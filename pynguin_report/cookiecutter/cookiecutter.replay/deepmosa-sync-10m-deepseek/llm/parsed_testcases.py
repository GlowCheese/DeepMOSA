####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_load_valid_json_with_cookiecutter_key. Retrieved 8/17 statements.
# Partially parsed test_load_valid_json_without_json_extension. Retrieved 7/16 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 6/16 statements.
# Partially parsed test_load_file_not_found. Retrieved 1/7 statements.
# Partially parsed test_load_with_path_object. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = var_0 / var_5
    var_7 = 'template'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = var_0 / var_5

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'template.json'
    var_4 = var_0 / var_3
    var_5 = 'template'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'nonexistent'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template.json'
    var_6 = var_0 / var_5
    var_7 = 'template'
    var_8 = module_0.load(var_2, var_7)
    var_9 = bool(var_8 == var_4)
    assert var_9 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_get_file_name_with_path_object_and_no_json_suffix. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_path_object_and_json_suffix. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_str_dir_and_no_json_suffix. Retrieved 5/6 statements.
# Partially parsed test_get_file_name_with_str_dir_and_json_suffix. Retrieved 5/6 statements.
# Partially parsed test_get_file_name_with_empty_template_name. Retrieved 3/6 statements.
# Partially parsed test_get_file_name_with_template_name_already_having_dot_json. Retrieved 5/6 statements.
# Partially parsed test_get_file_name_with_template_name_having_other_suffix. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = 'template.json'

def test_case_0():
    var_0 = '/another/dir'
    var_1 = [var_0]
    var_2 = 'template.json'
    var_3 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/str/dir'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = '/str/dir'
    var_4 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/str/dir'
    var_1 = 'my_template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = '/str/dir'
    var_4 = 'my_template.json'

def test_case_0():
    var_0 = '/empty'
    var_1 = [var_0]
    var_2 = ''
    var_3 = '.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/test'
    var_1 = 'data.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    var_3 = '/test'
    var_4 = 'data.json'

def test_case_0():
    var_0 = '/mixed'
    var_1 = [var_0]
    var_2 = 'file.txt'
    var_3 = 'file.txt.json'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/12 statements.


import json as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    var_3 = 'fake_dir'
    var_4 = [var_3]
    var_5 = 'fake_template'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_creates_directory_and_file. Retrieved 9/16 statements.
# Partially parsed test_dump_with_existing_json_extension. Retrieved 9/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/invalid_root_path/test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = bool(True)
    assert var_9 is True



# Parsed testcases at query #5
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'fake_dir'
    var_1 = 'fake_template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_context_contains_cookiecutter_key. Retrieved 9/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = '/fake/dir'
    var_7 = [var_6]
    var_8 = 'template'
    var_9 = module_0.load(var_1, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_load_success. Retrieved 11/14 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/14 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = 'utf-8'
    var_12 = bool(var_10 == var_7)
    assert var_12 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = 'utf-8'
    var_12 = bool(var_10 == var_7)
    assert var_12 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'some_key'
    var_4 = 'some_value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_context_without_cookiecutter_key. Retrieved 4/13 statements.


import json as module_0

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = module_0.dumps(var_0, **var_1)
    var_3 = 'test_replay.json'
    var_4 = [var_3]
    var_5 = 'test_template'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'some_dir'
    var_1 = 'some_template'
    var_2 = 'not_cookiecutter'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #12
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'dummy_dir'
    var_7 = 'dummy_template'
    var_8 = module_0.load(var_6, var_7)
    var_9 = 'cookiecutter'
    var_10 = bool('cookiecutter' in var_8)
    assert var_10 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/cookiecutter-test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'some_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------




import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'dummy_dir'
    var_8 = 'dummy_template'
    var_9 = module_1.load(var_7, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_load_context_contains_cookiecutter_key. Retrieved 9/15 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = 'test_template'
    var_9 = module_0.load(var_1, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/10 statements.
# Partially parsed test_load_with_json_extension. Retrieved 9/11 statements.
# Partially parsed test_load_without_json_extension. Retrieved 9/11 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
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



# Parsed testcases at query #20
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 3/12 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'test_template.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_appends_json_suffix_only_if_missing. Retrieved 8/14 statements.


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
    var_8 = [var_0]
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_8 = [var_0]
    var_9 = f'{var_1}.json'
    var_10 = [var_0]

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
    var_8 = [var_0]
    var_9 = [var_0]



# Parsed testcases at query #26
#--------------------------




def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = var_0 not in var_2
    assert var_3 is False



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = 'fake_template'
    var_7 = module_1.load(var_5, var_6)
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'some_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 9/16 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'fake_dir'
    var_8 = [var_7]
    var_9 = 'fake_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_returns_context_with_cookiecutter_key. Retrieved 9/16 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'fake_dir'
    var_8 = [var_7]
    var_9 = 'fake_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/17 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'fake_template'
    var_3 = '/fake/dir/fake_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_load_success. Retrieved 11/15 statements.
# Partially parsed test_load_with_json_extension. Retrieved 11/15 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template.json'
    var_2 = '/tmp/test_template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True
    var_12 = 'utf-8'

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'test_template'
    var_2 = '/tmp/test_template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 7/13 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_dump_raises_error_when_cookiecutter_key_missing. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'other_key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_context_contains_cookiecutter_key. Retrieved 9/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'test_dir'
    var_7 = [var_6]
    var_8 = 'test_template'
    var_9 = module_0.load(var_1, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/16 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_get_file_name_with_path_and_no_json_suffix. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_with_path_and_json_suffix. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_with_empty_template_name. Retrieved 2/4 statements.
# Partially parsed test_get_file_name_with_template_name_having_other_suffix. Retrieved 2/4 statements.


def test_case_0():
    var_0 = '/some/dir'
    var_1 = [var_0]
    var_2 = 'template'

def test_case_0():
    var_0 = '/another/dir'
    var_1 = [var_0]
    var_2 = 'template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/str/dir'
    var_1 = 'my_template'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/str/dir/my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/str/another'
    var_1 = 'my_template.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/str/another/my_template.json'

def test_case_0():
    var_0 = '/empty'
    var_1 = [var_0]
    var_2 = ''

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/path'
    var_1 = 'file.json'
    var_2 = module_0.get_file_name(var_0, var_1)
    assert var_2 == '/path/file.json'

def test_case_0():
    var_0 = '/other'
    var_1 = [var_0]
    var_2 = 'file.txt'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 8/14 statements.


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
    var_8 = [var_0]
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'cookiecutter'
    var_3 = 'project'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = [var_0]
    var_9 = f'{var_1}.json'
    var_10 = [var_0]

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
    var_8 = [var_0]
    var_9 = [var_0]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_json_decode_error. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_decode_error. Retrieved 4/7 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 7/13 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'not_cookiecutter'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_load_success. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_suffix. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/11 statements.
# Partially parsed test_load_invalid_json. Retrieved 4/8 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
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
    var_0 = '/tmp/replay'
    var_1 = 'template.json'
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
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/replay'
    var_1 = 'template'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'test_dir/test_template.json'
    var_4 = [var_3]
    var_5 = {}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 11/20 statements.
# Partially parsed test_dump_raises_value_error_without_cookiecutter_key. Retrieved 4/11 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 6/17 statements.
# Partially parsed test_dump_writes_indented_json. Retrieved 7/16 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'subdir'
    var_6 = var_0 / var_5
    var_7 = 'my_template'
    var_8 = module_0.dump(var_6, var_7, var_4)
    var_9 = f'{var_7}.json'
    var_10 = var_6 / var_9

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'Test'
    var_2 = {var_0: var_1}
    var_3 = 'my_template'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Context is required to contain a cookiecutter key'

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'my_template.json'
    var_6 = bool(var_1)
    assert var_6 is True

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'template'
    var_6 = f'{var_5}.json'
    var_7 = '\n'
    var_8 = '  '



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'not_cookiecutter'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_invalid_json. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template.json'
    var_2 = '/tmp/test/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test'
    var_1 = 'template'
    var_2 = '/tmp/test/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_context_contains_cookiecutter_key. Retrieved 9/16 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = 'test_dir'
    var_8 = [var_7]
    var_9 = 'test_template'
    var_10 = module_1.load(var_1, var_9)
    var_11 = 'cookiecutter'
    var_12 = bool('cookiecutter' in var_10)
    assert var_12 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_load_contains_cookiecutter_key. Retrieved 9/14 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = lambda x: var_4
    var_6 = 'fake_dir'
    var_7 = [var_6]
    var_8 = 'fake_template'
    var_9 = module_0.load(var_1, var_8)
    var_10 = 'cookiecutter'
    var_11 = bool('cookiecutter' in var_9)
    assert var_11 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_load_context_missing_cookiecutter_key. Retrieved 7/13 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_dump_creates_directory_and_writes_file. Retrieved 9/15 statements.
# Partially parsed test_dump_handles_existing_json_suffix. Retrieved 9/13 statements.
# Partially parsed test_dump_handles_nested_directory_creation. Retrieved 12/20 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'nested/test/replay'
    var_1 = 'template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'template.json'
    var_9 = 'nested'
    var_10 = 'test'
    var_11 = 'replay'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_load_success. Retrieved 10/13 statements.
# Partially parsed test_load_with_json_extension. Retrieved 10/13 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 8/12 statements.
# Partially parsed test_load_invalid_json. Retrieved 5/9 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = {}
    var_9 = module_0.dumps(var_7, **var_8)
    var_10 = module_1.load(var_0, var_1)
    var_11 = bool(var_10 == var_7)
    assert var_11 is True

import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = module_0.dumps(var_5, **var_6)
    var_8 = module_1.load(var_0, var_1)
    var_9 = bool(False)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'invalid json'
    var_4 = module_0.load(var_0, var_1)
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_not_in_context. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/10 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/14 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/13 statements.


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

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_8 = 'test_template.json'

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
    var_8 = 'test_template.json'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 7/14 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = 'some_key'
    var_1 = 'some_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.dumps(var_2, **var_3)
    var_5 = 'fake_dir'
    var_6 = [var_5]
    var_7 = 'fake_template'
    var_8 = module_1.load(var_1, var_7)
    var_9 = bool(False)
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/14 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = '/fake/dir/test_template.json'
    var_4 = [var_3]
    var_5 = 'some_key'
    var_6 = 'some_value'
    var_7 = {var_5: var_6}
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'test_template.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}



# Parsed testcases at query #35
#--------------------------




import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_dump_raises_value_error_when_cookiecutter_not_in_context. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_template'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_load_success. Retrieved 9/11 statements.
# Partially parsed test_load_with_json_extension. Retrieved 9/11 statements.
# Partially parsed test_load_missing_cookiecutter. Retrieved 7/10 statements.
# Partially parsed test_load_json_decode_error. Retrieved 4/7 statements.


import json as module_0
import cookiecutter.replay as module_1

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
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
    var_0 = '/tmp'
    var_1 = 'template.json'
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
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = {}
    var_6 = module_0.dumps(var_4, **var_5)
    var_7 = module_1.load(var_0, var_1)
    var_8 = bool(False)
    assert var_8 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = 'invalid json'
    var_3 = module_0.load(var_0, var_1)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_load_success. Retrieved 9/12 statements.
# Partially parsed test_load_with_json_extension. Retrieved 9/12 statements.
# Partially parsed test_load_missing_cookiecutter_key. Retrieved 7/11 statements.
# Partially parsed test_load_with_path_object. Retrieved 8/13 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template.json'
    var_2 = '/tmp/template.json'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = module_0.load(var_0, var_1)
    var_9 = bool(var_8 == var_7)
    assert var_9 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'template'
    var_2 = '/tmp/template.json'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.load(var_0, var_1)
    var_7 = bool(False)
    assert var_7 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp'
    var_1 = 'nonexistent'
    var_2 = module_0.load(var_0, var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = '/tmp'
    var_1 = [var_0]
    var_2 = 'template'
    var_3 = '/tmp/template.json'
    var_4 = 'cookiecutter'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/9 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/12 statements.
# Partially parsed test_dump_appends_json_suffix_if_missing. Retrieved 9/11 statements.
# Partially parsed test_dump_does_not_append_json_suffix_if_present. Retrieved 9/11 statements.


import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = '/tmp/test_replay'
    var_1 = 'my_template.json'
    var_2 = 'cookiecutter'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = module_0.dump(var_0, var_1, var_6)
    var_8 = 'my_template.json'



# Parsed testcases at query #40
#--------------------------






# Parsed testcases at query #41
#--------------------------






# Parsed testcases at query #42
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_json_file. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 8/14 statements.


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
    var_8 = [var_0]
    var_9 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_8 = [var_0]
    var_9 = f'{var_1}.json'
    var_10 = [var_0]

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
    var_8 = [var_0]
    var_9 = [var_0]



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_load_raises_value_error_when_cookiecutter_key_missing. Retrieved 6/15 statements.


def test_case_0():
    var_0 = '/fake/dir'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = f'{var_2}.json'
    var_4 = 'some_key'
    var_5 = 'some_value'
    var_6 = {var_4: var_5}
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_dump_creates_file_with_utf8_encoding. Retrieved 9/17 statements.


import json as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = [var_0]
    var_2 = 'test_template'
    var_3 = 'cookiecutter'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = {var_3: var_6}
    var_8 = 2
    var_9 = {}
    var_10 = module_0.dumps(var_7, indent=var_8, **var_9)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_dump_creates_directory_if_not_exists. Retrieved 8/12 statements.
# Partially parsed test_dump_writes_correct_file. Retrieved 9/17 statements.
# Partially parsed test_dump_handles_template_name_with_json_extension. Retrieved 9/15 statements.
# Partially parsed test_dump_creates_file_with_proper_indentation. Retrieved 10/20 statements.


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
    var_8 = [var_0]
    var_9 = [var_0]

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
    var_8 = [var_0]
    var_9 = 'test_template.json'
    var_10 = [var_0]

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
    var_8 = [var_0]
    var_9 = 'test_template.json'
    var_10 = [var_0]

import cookiecutter.replay as module_0

def test_case_0():
    var_0 = 'test_replay'
    var_1 = 'test_template'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = module_0.dump(var_0, var_1, var_4)
    var_6 = bool(False)
    assert var_6 is True

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
    var_8 = [var_0]
    var_9 = 'test_template.json'
    var_10 = 0
    var_11 = [var_0]



