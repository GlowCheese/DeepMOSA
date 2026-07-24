####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_value2'
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = 'choice1'
    var_13 = 'choice2'
    var_14 = 'choice3'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_13}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_12, var_13]
    var_20 = {var_0: var_19}
    var_21 = 'invalid_choice'
    var_22 = {var_0: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = [var_12, var_13, var_14]
    var_25 = {var_23: var_24}
    var_26 = [var_12, var_14]
    var_27 = {var_23: var_26}
    var_28 = module_0.apply_overwrites_to_context(var_25, var_27)
    var_29 = [var_12, var_13]
    var_30 = {var_23: var_29}
    var_31 = [var_12, var_21]
    var_32 = {var_23: var_31}
    var_33 = module_0.apply_overwrites_to_context(var_30, var_32)
    var_34 = 'key1'
    var_35 = 'key2'
    var_36 = {var_34: var_2, var_35: var_3}
    var_37 = {var_33: var_36}
    var_38 = {var_34: var_5}
    var_39 = {var_33: var_38}
    var_40 = module_0.apply_overwrites_to_context(var_37, var_39)
    var_41 = {var_34: var_2}
    var_42 = {var_33: var_41}
    var_43 = {var_34: var_5, var_35: var_3}
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = True
    var_47 = {var_33: var_46}
    var_48 = 'yes'
    var_49 = {var_33: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = False
    var_52 = {var_33: var_51}
    var_53 = 'invalid_bool'
    var_54 = {var_33: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)
    var_56 = 'nested'
    var_57 = {var_34: var_2}
    var_58 = {var_56: var_57}
    var_59 = {var_55: var_58}
    var_60 = {var_34: var_5}
    var_61 = {var_56: var_60}
    var_62 = {var_55: var_61}
    var_63 = module_0.apply_overwrites_to_context(var_59, var_62)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp/test_output'
    var_4 = module_0.Environment()
    var_5 = '{{ cookiecutter.project_name }}'
    var_6 = False
    var_7 = True
    var_8 = ''
    var_9 = 'wrong_key'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = False
    var_13 = 'existing'
    var_14 = 'existing'
    var_15 = False



# Parsed testcases at query #3
#--------------------------


import jinja2.environment as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_var'
    var_2 = 'test_value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/test_output'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = 'test_dir'
    var_8 = '{{ test_var }}'
    var_9 = True
    var_10 = ''
    var_11 = '{{ undefined_var }}'
    var_12 = 'test_dir'
    var_13 = module_1.rmtree(var_2)
    var_14 = 'test_value'
    var_15 = module_1.rmtree(var_2)



# Parsed testcases at query #4
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_value2'
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = 'choice_var'
    var_13 = 'choice1'
    var_14 = 'choice2'
    var_15 = 'choice3'
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_12: var_16}
    var_18 = {var_12: var_14}
    var_19 = module_0.apply_overwrites_to_context(var_17, var_18)
    var_20 = [var_13, var_14]
    var_21 = {var_12: var_20}
    var_22 = 'invalid_choice'
    var_23 = {var_12: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = 'multi_choice'
    var_26 = [var_13, var_14, var_15]
    var_27 = {var_25: var_26}
    var_28 = [var_14, var_15]
    var_29 = {var_25: var_28}
    var_30 = module_0.apply_overwrites_to_context(var_27, var_29)
    var_31 = [var_13, var_14]
    var_32 = {var_25: var_31}
    var_33 = [var_13, var_22]
    var_34 = {var_25: var_33}
    var_35 = module_0.apply_overwrites_to_context(var_32, var_34)
    var_36 = 'bool_var'
    var_37 = True
    var_38 = {var_36: var_37}
    var_39 = 'yes'
    var_40 = {var_36: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_38, var_40)
    var_42 = False
    var_43 = {var_36: var_42}
    var_44 = 'invalid_bool'
    var_45 = {var_36: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = 'dict_var'
    var_48 = 'key1'
    var_49 = 'key2'
    var_50 = {var_48: var_2, var_49: var_3}
    var_51 = {var_47: var_50}
    var_52 = {var_48: var_5}
    var_53 = {var_47: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = {var_48: var_2}
    var_56 = {var_47: var_55}
    var_57 = {var_48: var_5, var_49: var_3}
    var_58 = {var_47: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)



# Parsed testcases at query #6
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = '{{ test }}_dir'
    var_8 = True
    var_9 = ''



# Parsed testcases at query #7
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = '*.bin'
    var_8 = [var_7]
    var_9 = {var_3: var_6, var_4: var_6, var_5: var_8}
    var_10 = {var_2: var_9}
    var_11 = '{{cookiecutter.project_slug}}'
    var_12 = True
    var_13 = 'test.txt'
    var_14 = 'Hello {{cookiecutter.project_name}}!'
    var_15 = 'binary.bin'
    var_16 = b'\x00\x01\x02\x03'
    var_17 = 'cookiecutter.generate.find_template'
    var_18 = 'cookiecutter.generate.create_env_with_context'
    var_19 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_20 = False
    var_21 = 'test'
    var_22 = {var_3: var_21}
    var_23 = {var_2: var_22}
    var_24 = True
    var_25 = False
    var_26 = module_0.generate_files(var_0, var_23, var_1, var_24, var_25, var_24, var_25)
    var_27 = False
    var_28 = True
    var_29 = module_0.generate_files(var_0, var_10, var_1, var_27, var_27, var_28, var_27)



# Parsed testcases at query #8
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = module_3.rmtree(var_0)



# Parsed testcases at query #9
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_value2'
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = 'choice1'
    var_13 = 'choice2'
    var_14 = 'choice3'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_13}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_12, var_13]
    var_20 = {var_0: var_19}
    var_21 = 'invalid_choice'
    var_22 = {var_0: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = [var_12, var_13, var_14]
    var_25 = {var_23: var_24}
    var_26 = [var_12, var_14]
    var_27 = {var_23: var_26}
    var_28 = module_0.apply_overwrites_to_context(var_25, var_27)
    var_29 = [var_12, var_13]
    var_30 = {var_23: var_29}
    var_31 = [var_12, var_21]
    var_32 = {var_23: var_31}
    var_33 = module_0.apply_overwrites_to_context(var_30, var_32)
    var_34 = 'nested1'
    var_35 = 'nested2'
    var_36 = {var_34: var_2, var_35: var_3}
    var_37 = {var_33: var_36}
    var_38 = {var_34: var_5}
    var_39 = {var_33: var_38}
    var_40 = module_0.apply_overwrites_to_context(var_37, var_39)
    var_41 = True
    var_42 = {var_33: var_41}
    var_43 = 'yes'
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = False
    var_47 = {var_33: var_46}
    var_48 = 'invalid_boolean'
    var_49 = {var_33: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_34: var_2}
    var_52 = {var_50: var_51}
    var_53 = {var_34: var_5, var_35: var_9}
    var_54 = {var_50: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'docs/readme.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'file.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False
    var_15 = {}
    var_16 = {var_0: var_15}
    var_17 = module_0.is_copy_only_path(var_7, var_16)
    assert var_17 is False
    var_18 = {}
    var_19 = module_0.is_copy_only_path(var_7, var_18)
    assert var_19 is False



# Parsed testcases at query #12
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = None
    var_12 = module_0.generate_context(var_0, var_11, var_10)
    var_13 = 'invalid json'
    var_14 = module_0.generate_context(var_0)



# Parsed testcases at query #13
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = {var_0: var_5, var_1: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_8, var_9)
    var_11 = 'nested_var1'
    var_12 = {var_11: var_2}
    var_13 = {var_0: var_12}
    var_14 = 'nested_var2'
    var_15 = {var_11: var_5, var_14: var_3}
    var_16 = {var_0: var_15}
    var_17 = module_0.apply_overwrites_to_context(var_13, var_16)
    var_18 = 'choice1'
    var_19 = 'choice2'
    var_20 = 'choice3'
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_0: var_21}
    var_23 = [var_19, var_20]
    var_24 = {var_0: var_23}
    var_25 = module_0.apply_overwrites_to_context(var_22, var_24)
    var_26 = [var_18, var_19, var_20]
    var_27 = {var_0: var_26}
    var_28 = 'choice4'
    var_29 = [var_28]
    var_30 = {var_0: var_29}
    var_31 = module_0.apply_overwrites_to_context(var_27, var_30)
    var_32 = [var_18, var_19, var_20]
    var_33 = {var_31: var_32}
    var_34 = {var_31: var_19}
    var_35 = module_0.apply_overwrites_to_context(var_33, var_34)
    var_36 = [var_18, var_19, var_20]
    var_37 = {var_31: var_36}
    var_38 = {var_31: var_28}
    var_39 = module_0.apply_overwrites_to_context(var_37, var_38)
    var_40 = True
    var_41 = {var_39: var_40}
    var_42 = 'yes'
    var_43 = {var_39: var_42}
    var_44 = module_0.apply_overwrites_to_context(var_41, var_43)
    var_45 = False
    var_46 = {var_39: var_45}
    var_47 = 'no'
    var_48 = {var_39: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_46, var_48)
    var_50 = {var_39: var_40}
    var_51 = 'invalid'
    var_52 = {var_39: var_51}
    var_53 = module_0.apply_overwrites_to_context(var_50, var_52)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'docs/index.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'file.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False
    var_15 = 'some_other_key'
    var_16 = 'value'
    var_17 = {var_15: var_16}
    var_18 = {var_0: var_17}
    var_19 = module_0.is_copy_only_path(var_7, var_18)
    assert var_19 is False
    var_20 = {}
    var_21 = module_0.is_copy_only_path(var_7, var_20)
    assert var_21 is False
    var_22 = 'static/*'
    var_23 = [var_22]
    var_24 = {var_1: var_23}
    var_25 = {var_0: var_24}
    var_26 = 'static/images/logo.png'
    var_27 = module_0.is_copy_only_path(var_26, var_25)
    assert var_27 is True
    var_28 = 'static/js/script.js'
    var_29 = module_0.is_copy_only_path(var_28, var_25)
    assert var_29 is True
    var_30 = 'dynamic/content.html'
    var_31 = module_0.is_copy_only_path(var_30, var_25)
    assert var_31 is False



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'test content'
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = 'output'
    var_10 = True
    var_11 = 'modified content'
    var_12 = 'hooks'
    var_13 = 'pre_gen_project.py'
    var_14 = "print('pre hook')"
    var_15 = 'post_gen_project.py'
    var_16 = "print('post hook')"
    var_17 = 'output_with_hooks'
    var_18 = "raise Exception('hook failed')"
    var_19 = 'output_failure'
    var_20 = True
    var_21 = 'output_failure'



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/mocks/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/mocks/nonexistent.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/valid-template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/mocks/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = 'README.md'
    var_9 = 'setup.py'



# Parsed testcases at query #18
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'test.txt'
    var_3 = 'Hello, {{ name }}!'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_copy_without_render'
    var_7 = '_new_lines'
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = False
    var_14 = 'binary.bin'
    var_15 = b'\x00\x01\x02\x03'
    var_16 = '*.bin'
    var_17 = True



# Parsed testcases at query #20
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test.txt'
    var_2 = 'cookiecutter'
    var_3 = '_copy_without_render'
    var_4 = '_new_lines'
    var_5 = []
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = module_3.rmtree(var_0)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'



# Parsed testcases at query #22
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = 'test_template.txt'
    var_14 = module_3.rmtree(var_0)



# Parsed testcases at query #23
#--------------------------


import jinja2.loaders as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'path/to/template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'path/to/output'
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = True
    var_11 = '.'
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = 'path/to/output/test_project'
    var_14 = True
    var_15 = 'dir1'
    var_16 = 'dir2'
    var_17 = [var_15, var_16]
    var_18 = 'file1.txt'
    var_19 = 'file2.txt'
    var_20 = [var_18, var_19]
    var_21 = (var_11, var_17, var_20)
    var_22 = []
    var_23 = 'file3.txt'
    var_24 = [var_23]
    var_25 = (var_15, var_22, var_24)
    var_26 = module_1.generate_files(var_0, var_5, var_6, var_7, var_8, var_9, var_10)
    assert var_26 == 'path/to/output/test_project'
    var_27 = 'template'
    var_28 = 'path/to/template'



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'test content'
    var_6 = 'output'
    var_7 = 'test'
    var_8 = 'file.txt'
    var_9 = 'template'
    var_10 = 'project_name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'output'
    var_15 = 'test'
    var_16 = True
    var_17 = 'template'
    var_18 = 'project_name'
    var_19 = 'test'
    var_20 = {var_18: var_19}
    var_21 = '{{cookiecutter.project_name}}'
    var_22 = 'test content'
    var_23 = 'output'
    var_24 = 'test'
    var_25 = 'existing content'
    assert var_25 == 'existing content'
    var_26 = True
    var_27 = 'template'
    var_28 = 'project_name'
    var_29 = 'test'
    var_30 = {var_28: var_29}
    var_31 = 'hooks'
    var_32 = 'print("pre hook")'
    var_33 = 'print("post hook")'
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'output'
    var_36 = True
    var_37 = 'test'
    var_38 = 'template'
    var_39 = 'project_name'
    var_40 = 'test'
    var_41 = {var_39: var_40}
    var_42 = '{{cookiecutter.project_name}}'
    var_43 = 'test content'
    var_44 = 'output'
    var_45 = 'cookiecutter'
    var_46 = 'project_name'
    var_47 = 'myproject'
    var_48 = {var_46: var_47}
    var_49 = {var_45: var_48}
    var_50 = 'myproject.txt'
    var_51 = 'template'
    var_52 = 'project_name'
    var_53 = '_copy_without_render'
    var_54 = 'test'
    var_55 = '*.bin'
    var_56 = [var_55]
    var_57 = {var_52: var_54, var_53: var_56}
    var_58 = '{{cookiecutter.project_name}}'
    var_59 = 'binary content'
    var_60 = 'output'
    var_61 = 'test'
    var_62 = 'file.bin'
    var_63 = 'template'
    var_64 = 'project_name'
    var_65 = 'test'
    var_66 = {var_64: var_65}
    var_67 = '{{cookiecutter.project_name}}'
    var_68 = 'output'
    var_69 = 'test'
    var_70 = False
    var_71 = 'template'
    var_72 = 'project_name'
    var_73 = 'test'
    var_74 = {var_72: var_73}
    var_75 = '{{cookiecutter.undefined_var}}'
    var_76 = 'output'



# Parsed testcases at query #25
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'project_name'
    var_6 = 'default_project'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_project'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #26
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)
    var_14 = 'non_existent_file.json'
    var_15 = module_0.generate_context(var_14)



# Parsed testcases at query #28
#--------------------------


def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'tests/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = '*.bin'
    var_8 = [var_7]
    var_9 = {var_3: var_6, var_4: var_6, var_5: var_8}
    var_10 = {var_2: var_9}
    var_11 = True
    var_12 = False
    var_13 = 'README.md'
    var_14 = 'binary_file.bin'



# Parsed testcases at query #29
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)
    var_14 = 'non_existent.json'
    var_15 = module_0.generate_context(var_14)



# Parsed testcases at query #30
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = 'cookiecutter.json'
    var_8 = {var_2: var_3}
    var_9 = module_0.dumps(var_8)
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = 'test_file.txt'
    var_12 = 'Hello, {{cookiecutter.project_name}}!'



# Parsed testcases at query #31
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'test_generate_file'
    var_1 = True
    var_2 = 'template'
    var_3 = 'project'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{ name }}!'
    var_6 = 'cookiecutter'
    var_7 = 'name'
    var_8 = '_copy_without_render'
    var_9 = '_new_lines'
    var_10 = []
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'World'
    var_14 = {var_6: var_12, var_7: var_13}
    var_15 = False



# Parsed testcases at query #33
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #34
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #35
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'choice_var'
    var_9 = 'option1'
    var_10 = 'option2'
    var_11 = 'option3'
    var_12 = [var_9, var_10, var_11]
    var_13 = {var_8: var_12}
    var_14 = {var_8: var_10}
    var_15 = module_0.apply_overwrites_to_context(var_13, var_14)
    var_16 = [var_9, var_10]
    var_17 = {var_8: var_16}
    var_18 = 'invalid_option'
    var_19 = {var_8: var_18}
    var_20 = module_0.apply_overwrites_to_context(var_17, var_19)
    var_21 = 'multi_choice'
    var_22 = 'a'
    var_23 = 'b'
    var_24 = 'c'
    var_25 = [var_22, var_23, var_24]
    var_26 = {var_21: var_25}
    var_27 = [var_22, var_24]
    var_28 = {var_21: var_27}
    var_29 = module_0.apply_overwrites_to_context(var_26, var_28)
    var_30 = [var_22, var_23]
    var_31 = {var_21: var_30}
    var_32 = 'd'
    var_33 = [var_22, var_32]
    var_34 = {var_21: var_33}
    var_35 = module_0.apply_overwrites_to_context(var_31, var_34)
    var_36 = 'bool_var'
    var_37 = True
    var_38 = {var_36: var_37}
    var_39 = 'y'
    var_40 = {var_36: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_38, var_40)
    var_42 = False
    var_43 = {var_36: var_42}
    var_44 = 'invalid'
    var_45 = {var_36: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = 'dict_var'
    var_48 = 'key1'
    var_49 = 'key2'
    var_50 = 'val1'
    var_51 = 'val2'
    var_52 = {var_48: var_50, var_49: var_51}
    var_53 = {var_47: var_52}
    var_54 = 'new_val1'
    var_55 = {var_48: var_54}
    var_56 = {var_47: var_55}
    var_57 = module_0.apply_overwrites_to_context(var_53, var_56)
    var_58 = {var_48: var_50}
    var_59 = {var_47: var_58}
    var_60 = {var_48: var_54, var_49: var_51}
    var_61 = {var_47: var_60}
    var_62 = module_0.apply_overwrites_to_context(var_59, var_61)
    var_63 = {var_46: var_2}
    var_64 = {var_46: var_5, var_1: var_3}
    var_65 = module_0.apply_overwrites_to_context(var_63, var_64)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = '{{ test }}'
    var_7 = 'rendered_dir'
    var_8 = {var_1: var_7}
    var_9 = module_0.Environment()
    var_10 = 'existing_dir'
    var_11 = {var_1: var_2}
    var_12 = module_0.Environment()
    var_13 = 'existing_dir'
    var_14 = True
    var_15 = 'existing_dir'
    var_16 = {var_1: var_2}
    var_17 = module_0.Environment()
    var_18 = ''
    var_19 = {var_1: var_2}
    var_20 = module_0.Environment()



# Parsed testcases at query #2
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = {var_1: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_8, var_9)
    var_11 = 'choice1'
    var_12 = 'choice2'
    var_13 = 'choice3'
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = {var_0: var_12}
    var_17 = module_0.apply_overwrites_to_context(var_15, var_16)
    var_18 = [var_11, var_12]
    var_19 = {var_0: var_18}
    var_20 = 'invalid_choice'
    var_21 = {var_0: var_20}
    var_22 = module_0.apply_overwrites_to_context(var_19, var_21)
    var_23 = [var_11, var_12, var_13]
    var_24 = {var_22: var_23}
    var_25 = [var_11, var_13]
    var_26 = {var_22: var_25}
    var_27 = module_0.apply_overwrites_to_context(var_24, var_26)
    var_28 = [var_11, var_12]
    var_29 = {var_22: var_28}
    var_30 = [var_11, var_20]
    var_31 = {var_22: var_30}
    var_32 = module_0.apply_overwrites_to_context(var_29, var_31)
    var_33 = True
    var_34 = {var_32: var_33}
    var_35 = 'yes'
    var_36 = {var_32: var_35}
    var_37 = module_0.apply_overwrites_to_context(var_34, var_36)
    var_38 = False
    var_39 = {var_32: var_38}
    var_40 = 'invalid_bool'
    var_41 = {var_32: var_40}
    var_42 = module_0.apply_overwrites_to_context(var_39, var_41)
    var_43 = 'subvar1'
    var_44 = 'subvar2'
    var_45 = {var_43: var_2, var_44: var_3}
    var_46 = {var_42: var_45}
    var_47 = {var_43: var_5}
    var_48 = {var_42: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_46, var_48)
    var_50 = {var_43: var_2}
    var_51 = {var_42: var_50}
    var_52 = {var_44: var_3}
    var_53 = {var_42: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = {var_42: var_2}
    var_56 = {var_1: var_3}
    var_57 = module_0.apply_overwrites_to_context(var_55, var_56, in_dictionary_variable=var_33)



# Parsed testcases at query #4
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'test_project'
    var_5 = "{{ cookiecutter.project_name.lower().replace(' ', '_') }}"
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '{{ cookiecutter.project_slug }}'
    var_9 = 'test.txt'
    var_10 = 'Hello {{ cookiecutter.project_name }}!'
    var_11 = 'Test Project'
    var_12 = {var_2: var_11}
    var_13 = 'output'
    var_14 = 'cookiecutter.hooks.run_hook_from_repo_dir'
    var_15 = True
    var_16 = 'bad_var'
    var_17 = 'test'
    var_18 = {var_16: var_17}
    var_19 = 'bad_output'
    var_20 = False
    var_21 = 'binary.bin'
    var_22 = b'\x00\x01\x02\x03'
    var_23 = '*.bin'
    var_24 = 'binary_output'



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'output'
    var_7 = True
    var_8 = 'test_project'
    var_9 = 'project_name'
    var_10 = 'custom_project'
    var_11 = {var_9: var_10}
    var_12 = 'existing.txt'
    var_13 = 'existing'
    var_14 = 'new'
    var_15 = {var_9: var_10}
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py'
    var_18 = "print('pre hook')"
    var_19 = 'post_gen_project.py'
    var_20 = "print('post hook')"
    var_21 = 'hook_project'
    var_22 = {var_9: var_21}
    var_23 = '{"invalid": "{{ undefined_var }}"}'
    var_24 = True



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = {var_0: var_5, var_1: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_8, var_9)
    var_11 = 'choice_var'
    var_12 = 'option1'
    var_13 = 'option2'
    var_14 = 'option3'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_11: var_15}
    var_17 = {var_11: var_13}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_12, var_13]
    var_20 = {var_11: var_19}
    var_21 = 'invalid_option'
    var_22 = {var_11: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = 'multi_var'
    var_25 = 'opt1'
    var_26 = 'opt2'
    var_27 = 'opt3'
    var_28 = [var_25, var_26, var_27]
    var_29 = {var_24: var_28}
    var_30 = [var_26, var_27]
    var_31 = {var_24: var_30}
    var_32 = module_0.apply_overwrites_to_context(var_29, var_31)
    var_33 = [var_25, var_26]
    var_34 = {var_24: var_33}
    var_35 = [var_27]
    var_36 = {var_24: var_35}
    var_37 = module_0.apply_overwrites_to_context(var_34, var_36)
    var_38 = 'bool_var'
    var_39 = True
    var_40 = {var_38: var_39}
    var_41 = 'yes'
    var_42 = {var_38: var_41}
    var_43 = module_0.apply_overwrites_to_context(var_40, var_42)
    var_44 = False
    var_45 = {var_38: var_44}
    var_46 = 'invalid'
    var_47 = {var_38: var_46}
    var_48 = module_0.apply_overwrites_to_context(var_45, var_47)
    var_49 = 'nested'
    var_50 = 'key1'
    var_51 = 'key2'
    var_52 = 'val1'
    var_53 = 'val2'
    var_54 = {var_50: var_52, var_51: var_53}
    var_55 = {var_49: var_54}
    var_56 = 'new_val1'
    var_57 = {var_50: var_56}
    var_58 = {var_49: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_55, var_58)
    var_60 = {var_50: var_52}
    var_61 = {var_49: var_60}
    var_62 = {var_50: var_56, var_51: var_53}
    var_63 = {var_49: var_62}
    var_64 = module_0.apply_overwrites_to_context(var_61, var_63)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'name'
    var_3 = 'default_name'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'version'
    var_7 = '2.0.0'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = 'tests/test-data/invalid.json'
    var_11 = module_0.generate_context(var_10)
    var_12 = 'tests/test-data/nonexistent.json'
    var_13 = module_0.generate_context(var_12)
    var_14 = 'tests/test-data/empty.json'
    var_15 = module_0.generate_context(var_14)



# Parsed testcases at query #8
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_copy_without_render'
    var_4 = '_new_lines'
    var_5 = []
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ name }}!'
    assert var_13 == 'Hello, World!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = 'test_binary.bin'
    var_16 = b'\x00\x01\x02\x03'
    assert var_16 == b'\x00\x01\x02\x03'
    assert var_16 == 'Hello, World!'
    var_17 = module_2.generate_file(var_0, var_15, var_8, var_11)
    var_18 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)
    var_19 = module_3.rmtree(var_0)



# Parsed testcases at query #9
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #10
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_file.txt'
    var_2 = 'cookiecutter'
    var_3 = '_copy_without_render'
    var_4 = '_new_lines'
    var_5 = []
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = module_3.rmtree(var_0)



# Parsed testcases at query #11
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'static/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'static/image.png'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'file.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'dynamic/image.png'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False
    var_15 = {}
    var_16 = module_0.is_copy_only_path(var_7, var_15)
    assert var_16 is False
    var_17 = 'subdir/file.txt'
    var_18 = module_0.is_copy_only_path(var_17, var_6)
    assert var_18 is True
    var_19 = 'subdir/static/file.png'
    var_20 = module_0.is_copy_only_path(var_19, var_6)
    assert var_20 is True



# Parsed testcases at query #12
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = False
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9, var_10)



# Parsed testcases at query #13
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ cookiecutter.project_name }}'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{ cookiecutter.project_name }}'
    var_10 = False
    var_11 = False
    var_12 = True
    var_13 = 'project_name'
    var_14 = ''
    var_15 = {var_13: var_14}
    var_16 = module_0.Environment()
    var_17 = '{{ cookiecutter.project_name }}'
    var_18 = 'project_name'
    var_19 = 'test_project'
    var_20 = {var_18: var_19}
    var_21 = module_0.Environment()
    var_22 = '{{ cookiecutter.nonexistent_var }}'



# Parsed testcases at query #14
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = [var_2, var_3]
    var_5 = {var_1: var_4}
    var_6 = {var_0: var_5}
    var_7 = 'file.txt'
    var_8 = module_0.is_copy_only_path(var_7, var_6)
    assert var_8 is True
    var_9 = 'docs/index.md'
    var_10 = module_0.is_copy_only_path(var_9, var_6)
    assert var_10 is True
    var_11 = 'file.py'
    var_12 = module_0.is_copy_only_path(var_11, var_6)
    assert var_12 is False
    var_13 = 'src/main.py'
    var_14 = module_0.is_copy_only_path(var_13, var_6)
    assert var_14 is False
    var_15 = {}
    var_16 = {var_0: var_15}
    var_17 = module_0.is_copy_only_path(var_7, var_16)
    assert var_17 is False
    var_18 = {}
    var_19 = module_0.is_copy_only_path(var_7, var_18)
    assert var_19 is False



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'test_project'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'template.txt'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'World'
    var_11 = {var_3: var_9, var_4: var_10}

def test_case_0():
    var_0 = 'test_project'
    var_1 = b'\x00\x01\x02\x03\x04'
    var_2 = 'binary.bin'
    var_3 = 'cookiecutter'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = []
    var_7 = '\n'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = {var_3: var_8}

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'template.txt'
    var_3 = 'Existing content'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_copy_without_render'
    var_7 = '_new_lines'
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = True

def test_case_0():
    var_0 = 'test_project'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'template.txt'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = ''
    var_11 = {var_3: var_9, var_4: var_10}



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'tests/mock_repo'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/output'
    var_7 = module_0.generate_files(var_0, var_5, var_6)
    var_8 = True
    var_9 = module_0.generate_files(var_0, var_5, var_6, var_8)
    var_10 = module_0.generate_files(var_0, var_5, var_6, skip_if_file_exists=var_8)
    var_11 = module_0.generate_files(var_0, var_5, var_6, accept_hooks=var_8)
    var_12 = module_0.generate_files(var_0, var_5, var_6, keep_project_on_failure=var_8)
    var_13 = module_1.rmtree(var_6)



# Parsed testcases at query #17
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'extra_project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/test-fixtures/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/test-fixtures/missing.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #18
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_value2'
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = 'choice1'
    var_13 = 'choice2'
    var_14 = 'choice3'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_13}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_12, var_13]
    var_20 = {var_0: var_19}
    var_21 = 'invalid_choice'
    var_22 = {var_0: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = [var_12, var_13, var_14]
    var_25 = {var_23: var_24}
    var_26 = [var_12, var_14]
    var_27 = {var_23: var_26}
    var_28 = module_0.apply_overwrites_to_context(var_25, var_27)
    var_29 = [var_12, var_13]
    var_30 = {var_23: var_29}
    var_31 = [var_12, var_21]
    var_32 = {var_23: var_31}
    var_33 = module_0.apply_overwrites_to_context(var_30, var_32)
    var_34 = 'key1'
    var_35 = 'key2'
    var_36 = {var_34: var_2, var_35: var_3}
    var_37 = {var_33: var_36}
    var_38 = {var_34: var_5}
    var_39 = {var_33: var_38}
    var_40 = module_0.apply_overwrites_to_context(var_37, var_39)
    var_41 = True
    var_42 = {var_33: var_41}
    var_43 = 'yes'
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = False
    var_47 = {var_33: var_46}
    var_48 = 'no'
    var_49 = {var_33: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_33: var_41}
    var_52 = 'invalid_bool'
    var_53 = {var_33: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = {var_34: var_2}
    var_56 = {var_54: var_55}
    var_57 = {var_34: var_5, var_35: var_9}
    var_58 = {var_54: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_value2'
    var_10 = {var_0: var_5, var_1: var_9}
    var_11 = module_0.apply_overwrites_to_context(var_8, var_10)
    var_12 = 'choice1'
    var_13 = 'choice2'
    var_14 = 'choice3'
    var_15 = [var_12, var_13, var_14]
    var_16 = {var_0: var_15}
    var_17 = {var_0: var_13}
    var_18 = module_0.apply_overwrites_to_context(var_16, var_17)
    var_19 = [var_12, var_13]
    var_20 = {var_0: var_19}
    var_21 = 'invalid_choice'
    var_22 = {var_0: var_21}
    var_23 = module_0.apply_overwrites_to_context(var_20, var_22)
    var_24 = [var_12, var_13, var_14]
    var_25 = {var_23: var_24}
    var_26 = [var_12, var_14]
    var_27 = {var_23: var_26}
    var_28 = module_0.apply_overwrites_to_context(var_25, var_27)
    var_29 = [var_12, var_13]
    var_30 = {var_23: var_29}
    var_31 = [var_12, var_21]
    var_32 = {var_23: var_31}
    var_33 = module_0.apply_overwrites_to_context(var_30, var_32)
    var_34 = True
    var_35 = {var_33: var_34}
    var_36 = 'yes'
    var_37 = {var_33: var_36}
    var_38 = module_0.apply_overwrites_to_context(var_35, var_37)
    var_39 = False
    var_40 = {var_33: var_39}
    var_41 = 'invalid_bool'
    var_42 = {var_33: var_41}
    var_43 = module_0.apply_overwrites_to_context(var_40, var_42)
    var_44 = 'nested1'
    var_45 = 'nested2'
    var_46 = {var_44: var_2, var_45: var_3}
    var_47 = {var_43: var_46}
    var_48 = {var_44: var_5}
    var_49 = {var_43: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_44: var_2}
    var_52 = {var_43: var_51}
    var_53 = {var_44: var_5, var_45: var_9}
    var_54 = {var_43: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)



# Parsed testcases at query #20
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'var1'
    var_1 = 'var2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value1'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = {var_0: var_5, var_1: var_3}
    var_10 = module_0.apply_overwrites_to_context(var_8, var_9)
    var_11 = 'choice1'
    var_12 = 'choice2'
    var_13 = 'choice3'
    var_14 = [var_11, var_12, var_13]
    var_15 = {var_0: var_14}
    var_16 = {var_0: var_12}
    var_17 = module_0.apply_overwrites_to_context(var_15, var_16)
    var_18 = [var_11, var_12]
    var_19 = {var_0: var_18}
    var_20 = {var_0: var_13}
    var_21 = module_0.apply_overwrites_to_context(var_19, var_20)
    var_22 = [var_11, var_12, var_13]
    var_23 = {var_21: var_22}
    var_24 = [var_12, var_13]
    var_25 = {var_21: var_24}
    var_26 = module_0.apply_overwrites_to_context(var_23, var_25)
    var_27 = [var_11, var_12]
    var_28 = {var_21: var_27}
    var_29 = [var_11, var_13]
    var_30 = {var_21: var_29}
    var_31 = module_0.apply_overwrites_to_context(var_28, var_30)
    var_32 = 'subvar1'
    var_33 = 'subvar2'
    var_34 = 'subvalue1'
    var_35 = 'subvalue2'
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = {var_31: var_36}
    var_38 = 'new_subvalue1'
    var_39 = {var_32: var_38}
    var_40 = {var_31: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_37, var_40)
    var_42 = {var_32: var_34}
    var_43 = {var_31: var_42}
    var_44 = {var_32: var_38, var_33: var_35}
    var_45 = {var_31: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = True
    var_48 = {var_31: var_47}
    var_49 = 'yes'
    var_50 = {var_31: var_49}
    var_51 = module_0.apply_overwrites_to_context(var_48, var_50)
    var_52 = {var_31: var_47}
    var_53 = 'invalid'
    var_54 = {var_31: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)
    var_56 = 'subsubvar1'
    var_57 = 'value'
    var_58 = {var_56: var_57}
    var_59 = {var_32: var_58}
    var_60 = {var_55: var_59}
    var_61 = 'new_value'
    var_62 = {var_56: var_61}
    var_63 = {var_32: var_62}
    var_64 = {var_55: var_63}
    var_65 = module_0.apply_overwrites_to_context(var_60, var_64)



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'project_name'
    assert var_2 == 'Hello test_project!'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = 'Hello {{ cookiecutter.project_name }}!'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = 'test_project'
    var_8 = 'test.txt'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = 'Hello {{ cookiecutter.project_name }}!'
    var_15 = '{{cookiecutter.project_name}}'
    var_16 = True
    var_17 = 'test_project'
    var_18 = 'test.txt'
    var_19 = 'template'
    var_20 = 'output'
    var_21 = 'project_name'
    var_22 = 'test_project'
    var_23 = {var_21: var_22}
    var_24 = 'Hello {{ cookiecutter.project_name }}!'
    var_25 = '{{cookiecutter.project_name}}'
    var_26 = 'Modified content'
    assert var_26 == 'Modified content'
    var_27 = True
    var_28 = 'template'
    var_29 = 'output'
    var_30 = 'project_name'
    var_31 = 'test_project'
    var_32 = {var_30: var_31}
    var_33 = 'hooks'
    var_34 = 'import sys; sys.exit(1)'
    var_35 = 'Hello {{ cookiecutter.project_name }}!'
    var_36 = '{{cookiecutter.project_name}}'
    var_37 = False
    var_38 = 'test_project'
    var_39 = 'test.txt'
    var_40 = 'template'
    var_41 = 'output'
    var_42 = 'project_name'
    var_43 = 'test_project'
    var_44 = {var_42: var_43}
    var_45 = 'hooks'
    var_46 = 'import sys; sys.exit(1)'
    var_47 = 'Hello {{ cookiecutter.project_name }}!'
    var_48 = '{{cookiecutter.project_name}}'
    var_49 = True
    var_50 = 'test_project'
    var_51 = 'template'
    var_52 = 'output'
    var_53 = 'project_name'
    assert var_53 == 'Hello custom_project!'
    var_54 = 'default_project'
    var_55 = {var_53: var_54}
    var_56 = 'Hello {{ cookiecutter.project_name }}!'
    var_57 = '{{cookiecutter.project_name}}'
    var_58 = 'project_name'
    var_59 = 'custom_project'
    var_60 = {var_58: var_59}
    var_61 = 'test.txt'



# Parsed testcases at query #22
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'project_name'
    var_9 = 'default_project'
    var_10 = {var_8: var_9}
    var_11 = 'cookiecutter.json'
    var_12 = 'project_name'
    var_13 = 'test_project'
    var_14 = {var_12: var_13}
    var_15 = 'project_name'
    var_16 = 'extra_project'
    var_17 = {var_15: var_16}
    var_18 = 'cookiecutter.json'
    var_19 = '{invalid json}'
    var_20 = 'non_existent_file.json'
    var_21 = module_0.generate_context(var_20)



# Parsed testcases at query #23
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/basic-template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test-project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'tests/test-outputs'
    var_7 = 'README.md'
    var_8 = 'src'
    var_9 = 'test_project'
    var_10 = True
    var_11 = 'tests/test-templates/template-with-hooks'
    var_12 = 'hook_output.txt'
    var_13 = 'tests/test-templates/template-with-error'
    var_14 = True
    var_15 = 'template-with-error'
    var_16 = False
    var_17 = 'tests/test-templates/template-with-copy-only'
    var_18 = {var_1: var_3}
    var_19 = 'copy_only.txt'
    var_20 = 'rendered.txt'
    var_21 = 'tests/test-templates/template-with-binary'
    var_22 = 'binary_file.bin'
    var_23 = 'tests/test-templates/template-with-empty-dir'
    var_24 = False



# Parsed testcases at query #24
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_0.generate_context(var_0)
    var_5 = 'key'
    var_6 = 'default_value'
    var_7 = {var_5: var_6}
    var_8 = module_0.generate_context(var_0, var_7)
    var_9 = 'extra_value'
    var_10 = {var_5: var_9}
    var_11 = module_0.generate_context(var_0, extra_context=var_10)
    var_12 = 'invalid json'
    var_13 = module_0.generate_context(var_0)



# Parsed testcases at query #25
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'name'
    var_4 = 'test'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = True
    var_11 = 'Hello, {{ cookiecutter.name }}!'
    var_12 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_13 = module_3.rmtree(var_0)



# Parsed testcases at query #26
#--------------------------


def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Hello, {{ name }}!'
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_copy_without_render'
    var_7 = '_new_lines'
    var_8 = 'World'
    var_9 = []
    var_10 = '\n'
    var_11 = {var_5: var_8, var_6: var_9, var_7: var_10}
    var_12 = {var_4: var_11}
    var_13 = False

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.bin'
    var_3 = b'\x00\x01\x02\x03'
    var_4 = 'cookiecutter'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = False

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Hello, {{ name }}!'
    var_4 = 'Existing content'
    var_5 = 'cookiecutter'
    var_6 = 'name'
    var_7 = '_copy_without_render'
    var_8 = '_new_lines'
    var_9 = 'World'
    var_10 = []
    var_11 = '\n'
    var_12 = {var_6: var_9, var_7: var_10, var_8: var_11}
    var_13 = {var_5: var_12}
    var_14 = True

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Hello, {{ undefined_var }}!'
    var_4 = 'cookiecutter'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = False



