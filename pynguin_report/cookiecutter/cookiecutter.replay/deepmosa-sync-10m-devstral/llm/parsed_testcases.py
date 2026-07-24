####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_file_with_correct_content. Retrieved 9/13 statements.


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
    var_8 = 'test_template.json'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_dump_raises_valueerror_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test-template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #4
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_missing_key'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/12 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/12 statements.


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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 8/12 statements.


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
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

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



# Parsed testcases at query #11
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_without_cookiecutter_key. Retrieved 5/10 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_requires_cookiecutter_key_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'some_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #18
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'nonexistent_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #21
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'key': 'value'}})
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_with_valid_json_and_cookiecutter_key. Retrieved 9/12 statements.
# Partially parsed test_load_with_valid_json_without_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_with_template_name_without_json_suffix. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/14 statements.
# Partially parsed test_load_with_json_file_missing_cookiecutter_key. Retrieved 8/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'valid_template'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == {'cookiecutter': {'key': 'value'}})
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'invalid_template'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    var_0 = {}
    var_1 = 'cookiecutter'
    var_2 = bool('cookiecutter' not in var_0)
    assert var_2 is True



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #30
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #31
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



# Parsed testcases at query #32
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'test.json'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = '{"cookiecutter": {}}'
    var_4 = 'utf-8'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'some_template'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = str(var_0)
    assert var_6 == 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dump_without_cookiecutter_key. Retrieved 5/7 statements.


def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'test'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_file_exists_and_readable. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'path/to/existing/dir'
    var_1 = [var_0]
    var_2 = 'valid_template'
    var_3 = '{"cookiecutter": {}}'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_with_valid_json. Retrieved 9/12 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_raises_value_error_without_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_no_ext'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_invalid'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #39
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #40
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_load_predicate_false. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 'nonexistent_directory'
    var_1 = 'test_template'
    var_2 = var_0 / var_1



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #43
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'key': 'value'}})
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_dir'
    var_1 = 'valid_template.json'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'key': 'value'}})
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'key': 'value'}})
    assert var_3 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_replay_file_opens_successfully. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'valid_path'
    var_1 = [var_0]
    var_2 = 'valid_template'
    var_3 = 'utf-8'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_object. Retrieved 2/4 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/template.json'

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = [var_0]
    var_2 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/dir'
    var_1 = 'subdir/template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/to/dir/subdir/template.json'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/14 statements.


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
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #4
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = {}
    var_8 = module_0.dumps(var_6, **var_7)
    var_9 = module_1.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 3/4 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_path'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/14 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 10/14 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 8/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == {'cookiecutter': {'key': 'value'}})
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == {'cookiecutter': {'key': 'value'}})
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_missing_key'
    var_2 = True
    var_3 = f'{var_1}.json'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #7
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #8
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #9
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'test'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #10
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #11
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'some_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_dump_missing_cookiecutter_key. Retrieved 6/9 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_file_not_found. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'nonexistent_file.json'
    var_1 = [var_0]



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_predicate_evaluates_to_false.




# Parsed testcases at query #15
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #16
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_replay_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'dummy_replay.json'
    var_1 = [var_0]
    var_2 = '{"cookiecutter": {}}'
    var_3 = 'utf-8'
    var_4 = 'template'



# Parsed testcases at query #18
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/15 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 8/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.get_file_name(var_0, var_1)
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #21
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 10/19 statements.
# Partially parsed test_load_with_json_file_without_cookiecutter_key. Retrieved 8/18 statements.
# Partially parsed test_load_with_json_file_without_json_extension. Retrieved 10/19 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = f'{var_1}.json'
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = f'{var_1}.json'
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_extension'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = f'{var_1}.json'
    var_9 = module_0.load(var_0, var_1)
    var_10 = bool(var_9 == var_6)
    assert var_10 is True



# Parsed testcases at query #23
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test-template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #24
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #25
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'valid_dir'
    var_1 = 'valid_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(var_2 == {'cookiecutter': {'key': 'value'}})
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'invalid_dir'
    var_1 = 'invalid_template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'nonexistent_template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = True
    var_5 = '{"key": "value"}'
    var_6 = 'utf-8'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_no_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path/to/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



# Parsed testcases at query #28
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
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/12 statements.
# Partially parsed test_dump_raises_value_error_if_no_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_json_suffix_in_template_name. Retrieved 10/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)
    var_9 = '.json'



# Parsed testcases at query #30
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'cookiecutter'
    var_4 = bool('cookiecutter' in var_2)
    assert var_4 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/12 statements.
# Partially parsed test_dump_uses_template_name_with_json_suffix. Retrieved 9/11 statements.


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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_dump_raises_value_error_when_context_missing_cookiecutter_key. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #33
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'dummy_path'
    var_1 = 'dummy_template'
    var_2 = 'other_key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 7/10 statements.


def test_case_0():
    var_0 = '/tmp/test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.


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
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'path/to/replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/13 statements.
# Partially parsed test_dump_raises_value_error_if_context_missing_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_template_name_with_json_suffix. Retrieved 9/13 statements.


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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'test_template.json'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_opens_file_with_utf8_encoding. Retrieved 5/7 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'dummy_dir'
    var_1 = 'dummy_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = 'dummy_file'
    var_4 = 'utf-8'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_dump_raises_value_error_when_context_lacks_cookiecutter_key. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test'
    var_1 = [var_0]
    var_2 = 'test-template'
    var_3 = 'some_key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dump_creates_replay_dir_and_writes_json_file. Retrieved 9/14 statements.


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
    var_8 = 'test_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_dump_creates_replay_file. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_content. Retrieved 9/11 statements.


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



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'cookiecutter'
    var_6 = bool('cookiecutter' in var_4)
    assert var_6 is True



# Parsed testcases at query #46
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.load(var_0, var_1)
    var_8 = bool(var_7 == var_6)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.load(var_0, var_1)



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_json. Retrieved 9/14 statements.


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
    var_8 = f'{var_1}.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_correct_context. Retrieved 7/13 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}



# Parsed testcases at query #49
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #50
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nonexistent_dir'
    var_1 = 'template_name'
    var_2 = module_0.load(var_0, var_1)



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_dump_creates_file_with_correct_content. Retrieved 10/16 statements.


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
    var_9 = [var_0]
    var_10 = [var_8]
    var_11 = True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_load_with_valid_json_and_cookiecutter_key. Retrieved 9/12 statements.
# Partially parsed test_load_with_valid_json_without_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_with_json_file_without_extension. Retrieved 9/12 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_dump_creates_replay_file_with_context. Retrieved 9/12 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 6/9 statements.
# Partially parsed test_dump_handles_json_suffix_in_template_name. Retrieved 9/12 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = 'Context is required to contain a cookiecutter key'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay_dir'
    var_1 = 'test_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = module_0.get_file_name(var_0, var_1)



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_load_with_valid_json_file. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_file_without_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_with_invalid_json_file. Retrieved 5/9 statements.
# Partially parsed test_load_with_missing_cookiecutter_key. Retrieved 7/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_without_suffix'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_6)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_invalid'
    var_2 = True
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template_missing_key'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.load(var_0, var_1)



