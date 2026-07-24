####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = '/tmp/replay'
    var_4 = 'test_template'
    var_5 = '/tmp/replay/test_template.json'
    var_6 = module_0.get_file_name(var_3, var_4)
    var_7 = 'test_template.json'
    var_8 = '/tmp/replay/test_template.json'
    var_9 = module_0.get_file_name(var_3, var_7)
    var_10 = '/tmp/replay'
    var_11 = 'test_template.json'
    var_12 = '/tmp/replay/test_template.json'
    var_13 = module_0.get_file_name(var_10, var_11)



# Parsed testcases at query #2
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'test_template'
    var_2 = '/tmp/replay/test_template.json'
    var_3 = '/tmp/replay'
    var_4 = 'test_template'
    var_5 = '/tmp/replay/test_template.json'
    var_6 = module_0.get_file_name(var_3, var_4)
    var_7 = 'test_template.json'
    var_8 = '/tmp/replay/test_template.json'
    var_9 = module_0.get_file_name(var_3, var_7)
    var_10 = '/tmp/replay'
    var_11 = 'test_template.json'
    var_12 = '/tmp/replay/test_template.json'
    var_13 = module_0.get_file_name(var_10, var_11)



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'test_project'
    var_5 = 'test_project_slug'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}



# Parsed testcases at query #5
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #6
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = {var_3: var_4}
    var_10 = module_0.dump(var_0, var_1, var_9)
    var_11 = 'test_template.json'
    var_12 = module_0.dump(var_0, var_11, var_6)
    var_13 = module_0.get_file_name(var_0, var_11)



# Parsed testcases at query #8
#--------------------------


def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #9
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #10
#--------------------------




# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 2



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #14
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #15
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #16
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #18
#--------------------------




# Parsed testcases at query #19
#--------------------------




# Parsed testcases at query #20
#--------------------------


import cookiecutter.replay as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = module_1.make_sure_path_exists(var_0)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------




# Parsed testcases at query #23
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



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


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #29
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'replay'
    var_5 = 2

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #31
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test.json'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = 2
    var_6 = 'test'

def test_case_0():
    var_0 = 'replay'
    var_1 = 'nonexistent'



# Parsed testcases at query #35
#--------------------------




# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2
    var_8 = 'test_template.json'
    var_9 = 2
    var_10 = 'invalid_template'
    var_11 = 'invalid_key'
    var_12 = 'value'
    var_13 = {var_11: var_12}
    var_14 = 2



# Parsed testcases at query #37
#--------------------------




# Parsed testcases at query #38
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #39
#--------------------------




# Parsed testcases at query #40
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #41
#--------------------------




# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #43
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = {var_3: var_4}



# Parsed testcases at query #45
#--------------------------




# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #47
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.json'
    var_6 = 'test'
    var_7 = 'invalid'
    var_8 = 'context'
    var_9 = {var_7: var_8}
    var_10 = 'non_existent'



# Parsed testcases at query #49
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #50
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #51
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #52
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 2

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'author'
    var_2 = 'test_project'
    var_3 = 'test_author'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_template'
    var_6 = 2

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template.json'
    var_8 = 2



# Parsed testcases at query #54
#--------------------------




# Parsed testcases at query #55
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #57
#--------------------------




# Parsed testcases at query #58
#--------------------------




# Parsed testcases at query #59
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #61
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #63
#--------------------------




# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = {var_3: var_4}



# Parsed testcases at query #65
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #66
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #67
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #69
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 2



# Parsed testcases at query #70
#--------------------------




# Parsed testcases at query #71
#--------------------------




# Parsed testcases at query #72
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #73
#--------------------------




# Parsed testcases at query #74
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #75
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #76
#--------------------------




# Parsed testcases at query #77
#--------------------------




# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 2



# Parsed testcases at query #79
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #80
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 2



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #84
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #85
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #88
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = 'test_template'
    var_4 = 'replay'
    var_5 = 2



# Parsed testcases at query #89
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'replay'
    var_5 = 2



# Parsed testcases at query #90
#--------------------------




# Parsed testcases at query #91
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #92
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #93
#--------------------------




# Parsed testcases at query #94
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #95
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #96
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #97
#--------------------------




# Parsed testcases at query #98
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #99
#--------------------------




# Parsed testcases at query #100
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'other_data'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'some_value'
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = 'test_template'
    var_10 = 2
    var_11 = {var_1: var_7}
    var_12 = 2



# Parsed testcases at query #101
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #102
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #103
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2
    var_7 = 'test_template.json'
    var_8 = 2
    var_9 = {var_1: var_2}
    var_10 = 'invalid_template'
    var_11 = 2
    var_12 = 'invalid_template'



# Parsed testcases at query #104
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #105
#--------------------------


import cookiecutter.replay as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = module_1.make_sure_path_exists(var_0)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #106
#--------------------------




# Parsed testcases at query #107
#--------------------------




# Parsed testcases at query #108
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #109
#--------------------------




# Parsed testcases at query #110
#--------------------------




# Parsed testcases at query #111
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #112
#--------------------------




# Parsed testcases at query #113
#--------------------------




# Parsed testcases at query #114
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 2
    var_9 = 'other_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 2



# Parsed testcases at query #115
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #116
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #118
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #119
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #120
#--------------------------




# Parsed testcases at query #121
#--------------------------




# Parsed testcases at query #122
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #123
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'replay'
    var_5 = 2



# Parsed testcases at query #124
#--------------------------




# Parsed testcases at query #125
#--------------------------




# Parsed testcases at query #126
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'replay'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 2



# Parsed testcases at query #127
#--------------------------




# Parsed testcases at query #128
#--------------------------




# Parsed testcases at query #129
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #130
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #131
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #132
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #133
#--------------------------




# Parsed testcases at query #134
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #135
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #136
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'replay'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'replay'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'nonexistent_template'
    var_1 = 'replay'



# Parsed testcases at query #137
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #138
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #139
#--------------------------




# Parsed testcases at query #140
#--------------------------




# Parsed testcases at query #141
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #142
#--------------------------


def test_case_0():
    var_0 = 'non_existent_template'



# Parsed testcases at query #143
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #144
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #145
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #146
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #147
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #148
#--------------------------




# Parsed testcases at query #149
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #150
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #151
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #152
#--------------------------


def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #153
#--------------------------




# Parsed testcases at query #154
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #155
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #156
#--------------------------




# Parsed testcases at query #157
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #158
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #159
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #160
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #161
#--------------------------


def test_case_0():
    var_0 = 'non_existent_template'
    var_1 = 'replay'



# Parsed testcases at query #162
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #163
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #164
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #165
#--------------------------




# Parsed testcases at query #166
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #167
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #168
#--------------------------




# Parsed testcases at query #169
#--------------------------




# Parsed testcases at query #170
#--------------------------




# Parsed testcases at query #171
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #172
#--------------------------


def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #173
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #174
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #175
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #176
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #177
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #178
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_3}
    var_8 = 2



# Parsed testcases at query #179
#--------------------------




# Parsed testcases at query #180
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)



# Parsed testcases at query #181
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #182
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'replay'
    var_5 = 2



# Parsed testcases at query #183
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'nonexistent_template'



# Parsed testcases at query #184
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #185
#--------------------------




# Parsed testcases at query #186
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'
    var_7 = 2
    var_8 = {var_1: var_2}
    var_9 = 'bad_template'
    var_10 = 2
    var_11 = 'bad_template'



# Parsed testcases at query #187
#--------------------------




# Parsed testcases at query #188
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #189
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #190
#--------------------------


def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #191
#--------------------------




# Parsed testcases at query #192
#--------------------------




# Parsed testcases at query #193
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'test_template'
    var_4 = 2



# Parsed testcases at query #194
#--------------------------




# Parsed testcases at query #195
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #196
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'project_slug'
    var_3 = 'test_project'
    var_4 = 'test_project_slug'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 2



# Parsed testcases at query #197
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #198
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #199
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #200
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #201
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'replay'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = 2
    var_9 = 'invalid_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 2
    var_13 = 'non_existent'
    var_14 = 'non_existent'



# Parsed testcases at query #202
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #203
#--------------------------




# Parsed testcases at query #204
#--------------------------


def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #205
#--------------------------




# Parsed testcases at query #206
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #207
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 'replay'
    var_9 = 2



# Parsed testcases at query #208
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #209
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #210
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #211
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2



# Parsed testcases at query #212
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #213
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = True
    var_10 = module_0.dump(var_0, var_1, var_8)
    var_11 = module_0.load(var_0, var_1)
    var_12 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #214
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'nonexistent_template'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------




# Parsed testcases at query #4
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 2



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_3}
    var_8 = 'invalid_template'
    var_9 = 2
    var_10 = 'invalid_template'



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #8
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #11
#--------------------------




# Parsed testcases at query #12
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #18
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)
    var_10 = module_0.get_file_name(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #20
#--------------------------




# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 2



# Parsed testcases at query #22
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 2



# Parsed testcases at query #25
#--------------------------




# Parsed testcases at query #26
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #28
#--------------------------




# Parsed testcases at query #29
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #30
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 2



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #33
#--------------------------




# Parsed testcases at query #34
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #35
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = 'test_template.json'
    var_8 = 2
    var_9 = 'invalid_template'
    var_10 = 'invalid_key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 2



# Parsed testcases at query #36
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 2

def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #37
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = {var_3: var_4}



# Parsed testcases at query #38
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #39
#--------------------------




# Parsed testcases at query #40
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = 'other_key'
    var_13 = {var_12: var_4}
    var_14 = module_0.dump(var_0, var_1, var_13)



# Parsed testcases at query #41
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)



# Parsed testcases at query #42
#--------------------------




# Parsed testcases at query #43
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #44
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2
    var_7 = {var_2: var_3}
    var_8 = 2



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test.json'
    var_8 = 2
    var_9 = 'test'
    var_10 = 'invalid_key'
    var_11 = 'value'
    var_12 = {var_10: var_11}
    var_13 = 'invalid.json'
    var_14 = 2
    var_15 = 'invalid'



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #49
#--------------------------




# Parsed testcases at query #50
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #51
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 2



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #54
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #56
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #57
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #58
#--------------------------




# Parsed testcases at query #59
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #61
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #62
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}



# Parsed testcases at query #63
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #64
#--------------------------




# Parsed testcases at query #65
#--------------------------




# Parsed testcases at query #66
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #67
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #68
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #69
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #70
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #71
#--------------------------




# Parsed testcases at query #72
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #73
#--------------------------


def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #74
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #75
#--------------------------


import cookiecutter.replay as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = module_1.make_sure_path_exists(var_0)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'test_template.json'
    var_7 = {var_1: var_2}
    var_8 = 'test_template'



# Parsed testcases at query #77
#--------------------------




# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #79
#--------------------------


import cookiecutter.replay as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = module_1.make_sure_path_exists(var_0)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #80
#--------------------------




# Parsed testcases at query #81
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #82
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #83
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #84
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #85
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 2
    var_7 = 'test_template.json'
    var_8 = 2
    var_9 = 'invalid_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 'invalid_template'
    var_13 = 2
    var_14 = 'invalid_template'



# Parsed testcases at query #86
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 2
    var_9 = 'invalid_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = 2



# Parsed testcases at query #87
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}



# Parsed testcases at query #88
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #89
#--------------------------




# Parsed testcases at query #90
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #91
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #92
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #93
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #94
#--------------------------




# Parsed testcases at query #95
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #96
#--------------------------




# Parsed testcases at query #97
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #98
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #99
#--------------------------




# Parsed testcases at query #100
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #101
#--------------------------




# Parsed testcases at query #102
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #103
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 2

def test_case_0():
    var_0 = 'nonexistent_template'



# Parsed testcases at query #104
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #105
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #106
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'
    var_7 = 2



# Parsed testcases at query #107
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #108
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template.json'
    var_7 = 2



# Parsed testcases at query #109
#--------------------------


def test_case_0():
    var_0 = 'non_existent_template'



# Parsed testcases at query #110
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 'replay'
    var_9 = 2



# Parsed testcases at query #111
#--------------------------




# Parsed testcases at query #112
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #113
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'project_slug'
    var_3 = 'test_project'
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = {var_0: var_4}
    var_6 = 'test_template'
    var_7 = 2



# Parsed testcases at query #114
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #115
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #116
#--------------------------


def test_case_0():
    var_0 = 'test_template.json'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #117
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #118
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2
    var_8 = {var_2: var_3}
    var_9 = 'bad_template'
    var_10 = 2
    var_11 = 'bad_template'



# Parsed testcases at query #119
#--------------------------




# Parsed testcases at query #120
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_template'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}



# Parsed testcases at query #121
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 2



# Parsed testcases at query #122
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #123
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #124
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #125
#--------------------------




# Parsed testcases at query #126
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #127
#--------------------------




# Parsed testcases at query #128
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #129
#--------------------------




# Parsed testcases at query #130
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #131
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #132
#--------------------------




# Parsed testcases at query #133
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #134
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #135
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #136
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #137
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #138
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 2



# Parsed testcases at query #139
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #140
#--------------------------




# Parsed testcases at query #141
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #142
#--------------------------




# Parsed testcases at query #143
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #144
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test.json'
    var_6 = 'new_dir'
    var_7 = 'test'
    var_8 = 'invalid'
    var_9 = 'context'
    var_10 = {var_8: var_9}



# Parsed testcases at query #145
#--------------------------




# Parsed testcases at query #146
#--------------------------




# Parsed testcases at query #147
#--------------------------




# Parsed testcases at query #148
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = {var_3: var_4}
    var_13 = module_0.dump(var_0, var_1, var_12)



# Parsed testcases at query #149
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #150
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 'test_template.json'
    var_10 = module_0.dump(var_0, var_9, var_6)
    var_11 = module_0.get_file_name(var_0, var_9)
    var_12 = 'other_key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = module_0.dump(var_0, var_1, var_14)



# Parsed testcases at query #151
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #152
#--------------------------




# Parsed testcases at query #153
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #154
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = {var_3: var_4}
    var_10 = module_0.dump(var_0, var_1, var_9)



# Parsed testcases at query #155
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 'replay'
    var_9 = 2



# Parsed testcases at query #156
#--------------------------




# Parsed testcases at query #157
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = 'test_project_slug'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = 2



# Parsed testcases at query #158
#--------------------------




# Parsed testcases at query #159
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #160
#--------------------------




# Parsed testcases at query #161
#--------------------------


def test_case_0():
    var_0 = 'non_existent_template'



# Parsed testcases at query #162
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #163
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #164
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'project_slug'
    var_3 = 'test_project'
    var_4 = 'test_project_slug'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'test_template'
    var_8 = 'replay'
    var_9 = 2



# Parsed testcases at query #165
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'replay'
    var_5 = 2



# Parsed testcases at query #166
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #167
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #168
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #169
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #170
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)



# Parsed testcases at query #171
#--------------------------


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'replay'
    var_7 = 2



# Parsed testcases at query #172
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'test_author'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = module_0.dump(var_0, var_1, var_8)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'author'
    var_4 = 'test_project'
    var_5 = 'test_author'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)



# Parsed testcases at query #173
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 2



# Parsed testcases at query #174
#--------------------------




# Parsed testcases at query #175
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #176
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 2



# Parsed testcases at query #177
#--------------------------




# Parsed testcases at query #178
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.dump(var_0, var_1, var_6)
    var_9 = module_0.load(var_0, var_1)
    var_10 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #179
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #180
#--------------------------


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'test_template'
    var_6 = 'replay'
    var_7 = 'test_template.json'
    var_8 = {var_1: var_2}



# Parsed testcases at query #181
#--------------------------


def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}



# Parsed testcases at query #182
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #183
#--------------------------




# Parsed testcases at query #184
#--------------------------


import cookiecutter.replay as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.get_file_name(var_0, var_1)
    var_8 = True
    var_9 = 2
    var_10 = module_0.load(var_0, var_1)
    var_11 = 'test_template.json'
    var_12 = module_0.get_file_name(var_0, var_11)
    var_13 = 2
    var_14 = module_0.load(var_0, var_11)
    var_15 = {var_3: var_4}
    var_16 = 'invalid_template'
    var_17 = module_0.get_file_name(var_0, var_16)
    var_18 = 2
    var_19 = 'invalid_template'
    var_20 = module_0.load(var_0, var_19)
    var_21 = module_1.rmtree(var_0)



# Parsed testcases at query #185
#--------------------------


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'test_template.json'
    var_8 = module_0.dump(var_0, var_1, var_6)



