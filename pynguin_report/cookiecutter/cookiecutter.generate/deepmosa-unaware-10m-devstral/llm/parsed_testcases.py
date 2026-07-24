####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test_project"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'Hello, {{cookiecutter.project_name}}!'
    var_7 = True
    var_8 = 'test_project'
    var_9 = 'project_name'
    var_10 = 'another_project'
    var_11 = {var_9: var_10}
    var_12 = 'Existing content'
    var_13 = 'binary.bin'
    var_14 = b'\x00\x01\x02\x03'
    var_15 = '{"project_name": "test_project", "_copy_without_render": ["*.md"]}'
    var_16 = 'readme.md'
    var_17 = '# {{cookiecutter.project_name}}'



# Parsed testcases at query #2
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
    var_34 = 'subvar1'
    var_35 = 'subvar2'
    var_36 = 'subvalue1'
    var_37 = 'subvalue2'
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = {var_33: var_38}
    var_40 = 'new_subvalue1'
    var_41 = {var_34: var_40}
    var_42 = {var_33: var_41}
    var_43 = module_0.apply_overwrites_to_context(var_39, var_42)
    var_44 = {var_34: var_36}
    var_45 = {var_33: var_44}
    var_46 = 'new_subvalue2'
    var_47 = {var_34: var_40, var_35: var_46}
    var_48 = {var_33: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_45, var_48)
    var_50 = True
    var_51 = {var_33: var_50}
    var_52 = 'y'
    var_53 = {var_33: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = False
    var_56 = {var_33: var_55}
    var_57 = 'invalid_bool'
    var_58 = {var_33: var_57}
    var_59 = module_0.apply_overwrites_to_context(var_56, var_58)
    var_60 = {var_59: var_2}
    var_61 = {var_59: var_5}
    var_62 = module_0.apply_overwrites_to_context(var_60, var_61)



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
    var_43 = 'y'
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = False
    var_47 = {var_33: var_46}
    var_48 = 'invalid_bool'
    var_49 = {var_33: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_34: var_2}
    var_52 = {var_50: var_51}
    var_53 = {var_34: var_5, var_35: var_9}
    var_54 = {var_50: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_52, var_54)



# Parsed testcases at query #4
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = 'project_name'
    var_16 = 'test_project'
    var_17 = {var_15: var_16}
    var_18 = module_0.Environment()
    var_19 = var_14 / var_16
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_17}
    var_23 = True
    var_24 = 'project_name'
    var_25 = ''
    var_26 = {var_24: var_25}
    var_27 = module_0.Environment()
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'cookiecutter'
    var_30 = {var_29: var_26}
    var_31 = 'project_name'
    var_32 = 'author'
    var_33 = 'my_project'
    var_34 = 'test_author'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Environment()
    var_37 = '{{cookiecutter.project_name}}_{{cookiecutter.author}}'
    var_38 = 'cookiecutter'
    var_39 = {var_38: var_35}



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test_file.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'cookiecutter.generate.find_template'
    var_7 = 'cookiecutter.generate.create_env_with_context'
    var_8 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_9 = 'output'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = {var_10: var_13}
    var_15 = True
    var_16 = {var_11: var_12}
    var_17 = {var_10: var_16}
    var_18 = {var_11: var_12}
    var_19 = {var_10: var_18}
    var_20 = 'output_no_hooks'
    var_21 = False



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'test.json'
    var_3 = 'key'
    var_4 = 'new_value'
    var_5 = {var_3: var_4}
    var_6 = module_0.generate_context(var_2, var_5)
    var_7 = 'test.json'
    var_8 = 'key'
    var_9 = 'extra_value'
    var_10 = {var_8: var_9}
    var_11 = module_0.generate_context(var_7, extra_context=var_10)
    var_12 = 'test.json'
    var_13 = module_0.generate_context(var_12)
    var_14 = 'nonexistent.json'
    var_15 = module_0.generate_context(var_14)



# Parsed testcases at query #7
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'project_slug'
    var_7 = 'extra_slug'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = 'tests/mocks/invalid.json'
    var_11 = module_0.generate_context(var_10)
    var_12 = 'tests/mocks/nonexistent.json'
    var_13 = module_0.generate_context(var_12)



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_16 = {var_0: var_15}
    var_17 = module_0.is_copy_only_path(var_7, var_16)
    assert var_17 is False
    var_18 = {}
    var_19 = module_0.is_copy_only_path(var_7, var_18)
    assert var_19 is False



# Parsed testcases at query #10
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
    var_43 = 'nested1'
    var_44 = 'nested2'
    var_45 = {var_43: var_2, var_44: var_3}
    var_46 = {var_42: var_45}
    var_47 = {var_43: var_5}
    var_48 = {var_42: var_47}
    var_49 = module_0.apply_overwrites_to_context(var_46, var_48)
    var_50 = {var_43: var_2}
    var_51 = {var_42: var_50}
    var_52 = {var_43: var_5, var_44: var_3}
    var_53 = {var_42: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)



# Parsed testcases at query #11
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



# Parsed testcases at query #12
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = 'project'
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
    var_12 = 'Hello, {{ name }}!'
    var_13 = 'binary.bin'
    var_14 = b'binary data'
    var_15 = '*.bin'
    var_16 = True
    var_17 = ''
    var_18 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_19 = []
    var_20 = {var_3: var_19}
    var_21 = {var_2: var_20}
    var_22 = module_2.generate_file(var_0, var_1, var_21, var_11)



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'project'
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
    var_12 = 'binary.bin'
    var_13 = b'\x00\x01\x02\x03'
    var_14 = True
    var_15 = 'Line1\nLine2'
    var_16 = 'bad.txt'
    var_17 = '{% if %}'
    var_18 = []
    var_19 = {var_5: var_18}
    var_20 = {var_3: var_19}



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"project_name": "test"}'
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'file.txt'
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = 'override'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = True
    var_21 = 'template'
    var_22 = 'output'
    var_23 = 'cookiecutter.json'
    var_24 = '{"project_name": "test"}'
    var_25 = '{{cookiecutter.project_name}}'
    var_26 = 'file.txt'
    var_27 = 'new content'
    var_28 = True
    var_29 = 'test'
    var_30 = 'template'
    var_31 = 'output'
    var_32 = 'cookiecutter.json'
    var_33 = '{"project_name": "test"}'
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'hooks'
    var_36 = 'pre_gen_project.py'
    var_37 = var_26 / var_36
    var_38 = "print('pre hook')"
    var_39 = 'post_gen_project.py'
    var_40 = var_28 / var_39
    var_41 = "print('post hook')"
    var_42 = True
    var_43 = 'test'
    var_44 = 'template'
    var_45 = 'output'
    var_46 = 'cookiecutter.json'
    var_47 = '{"project_name": "test"}'
    var_48 = '{{cookiecutter.undefined_var}}'
    var_49 = True



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'output'
    var_7 = 'test'
    var_8 = 'template'
    var_9 = 'cookiecutter.json'
    var_10 = '{"project_name": "test"}'
    var_11 = '{{cookiecutter.project_name}}'
    var_12 = 'output'
    var_13 = 'test'
    var_14 = True
    var_15 = 'template'
    var_16 = 'cookiecutter.json'
    var_17 = '{"project_name": "test"}'
    var_18 = '{{cookiecutter.project_name}}'
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = 'output'
    var_22 = 'test'
    var_23 = 'existing'
    var_24 = True
    var_25 = 'template'
    var_26 = 'cookiecutter.json'
    var_27 = '{"project_name": "test"}'
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'hooks'
    var_30 = 'pre_gen_project.py'
    var_31 = var_19 / var_30
    var_32 = "print('pre')"
    var_33 = 'post_gen_project.py'
    var_34 = var_21 / var_33
    var_35 = "print('post')"
    var_36 = 'output'
    var_37 = True
    var_38 = 'test'
    var_39 = 'template'
    var_40 = 'cookiecutter.json'
    var_41 = '{"project_name": "test"}'
    var_42 = '{{cookiecutter.project_name}}'
    var_43 = 'binary.bin'
    var_44 = var_29 / var_43
    var_45 = b'\x00\x01\x02'
    var_46 = 'output'
    var_47 = var_32 / var_46
    var_48 = str(var_47)
    var_49 = module_0.generate_files(var_21, output_dir=var_48)
    var_50 = 'test'
    var_51 = var_47 / var_50
    var_52 = var_51 / var_43
    var_53 = var_47 / var_50
    assert var_53 == '# {{cookiecutter.project_name}}'
    var_54 = var_53 / var_43
    var_55 = 'template'
    var_56 = 'cookiecutter.json'
    var_57 = '{"project_name": "test", "_copy_without_render": ["*.md"]}'
    var_58 = '{{cookiecutter.project_name}}'
    var_59 = 'readme.md'
    var_60 = var_29 / var_59
    var_61 = '# {{cookiecutter.project_name}}'
    var_62 = 'output'
    var_63 = var_46 / var_62
    var_64 = str(var_63)
    var_65 = module_0.generate_files(var_48, output_dir=var_64)
    var_66 = 'test'
    var_67 = var_63 / var_66
    var_68 = var_67 / var_59
    var_69 = 'template'
    var_70 = 'cookiecutter.json'
    var_71 = '{"project_name": "test"}'
    var_72 = '{{cookiecutter.undefined_var}}'
    var_73 = 'output'
    var_74 = var_29 / var_73
    var_75 = str(var_74)
    var_76 = 'template'
    var_77 = 'cookiecutter.json'
    var_78 = '{"project_name": "test"}'
    var_79 = '{{cookiecutter.project_name}}'
    var_80 = 'output'
    var_81 = var_29 / var_80
    var_82 = 'test'
    var_83 = var_81 / var_82
    var_84 = str(var_81)



# Parsed testcases at query #17
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = {var_7: var_8}
    var_13 = {var_6: var_12}
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py'
    var_16 = "print('pre hook')"
    var_17 = 'post_gen_project.py'
    var_18 = "print('post hook')"
    var_19 = 'test_project2'
    var_20 = {var_7: var_19}
    var_21 = {var_6: var_20}
    var_22 = 'cookiecutter'
    var_23 = 'project_name'
    var_24 = '{{undefined_var}}'
    var_25 = {var_23: var_24}
    var_26 = {var_22: var_25}
    var_27 = 'binary.bin'
    var_28 = b'\x00\x01\x02'
    var_29 = '{"project_name": "test_project3", "_copy_without_render": ["*.bin"]}'
    var_30 = 'test_project3'
    var_31 = {var_7: var_30}
    var_32 = {var_6: var_31}



# Parsed testcases at query #18
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = 'test_project'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = {var_2: var_9}
    var_11 = '.'
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = True
    var_15 = 'Hello, {{ cookiecutter.project_name }}!'
    var_16 = module_2.generate_file(var_0, var_1, var_10, var_13)
    var_17 = module_3.rmtree(var_0)



# Parsed testcases at query #19
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'test.txt'
    var_10 = 'Hello, {{cookiecutter.project_name}}!'
    var_11 = module_0.generate_files(var_0, var_5, var_6)
    var_12 = module_1.rmtree(var_0)
    var_13 = module_1.rmtree(var_6)



# Parsed testcases at query #20
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = var_4 / var_6
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = False
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = True
    var_14 = 'project_name'
    var_15 = ''
    var_16 = {var_14: var_15}
    var_17 = module_0.Environment()
    var_18 = '{{cookiecutter.project_name}}'
    var_19 = 'project_name'
    var_20 = 'version'
    var_21 = 'my_project'
    var_22 = '1.0'
    var_23 = {var_19: var_21, var_20: var_22}
    var_24 = module_0.Environment()
    var_25 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'



# Parsed testcases at query #21
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.create_env_with_context'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.create_env_with_context'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = True
    var_16 = False

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.create_env_with_context'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = 'Existing content'
    var_16 = False
    var_17 = True

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.undefined_var}}!'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test_project'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'cookiecutter.generate.find_template'
    var_13 = 'cookiecutter.generate.create_env_with_context'
    var_14 = 'cookiecutter.generate.run_hook_from_repo_dir'
    var_15 = True
    var_16 = False



# Parsed testcases at query #22
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_var'
    var_2 = 'test_value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = ''
    var_8 = True



# Parsed testcases at query #23
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
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = 'test.txt'
    var_14 = False

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
    var_11 = 'test.bin'
    var_12 = False

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
    var_9 = []
    var_10 = '\n'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'World'
    var_13 = {var_5: var_11, var_6: var_12}
    var_14 = 'test.txt'
    var_15 = True

def test_case_0():
    var_0 = 'project'
    var_1 = 'template'
    var_2 = 'test.txt'
    var_3 = 'Hello, {{ undefined_variable }}!'
    var_4 = 'cookiecutter'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {var_4: var_9}
    var_11 = 'test.txt'
    var_12 = False



# Parsed testcases at query #24
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
    var_15 = 'test_binary.bin'
    var_16 = b'\x00\x01\x02\x03'
    var_17 = module_2.generate_file(var_0, var_15, var_8, var_11)
    var_18 = module_3.rmtree(var_0)



# Parsed testcases at query #26
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
    var_19 = 'invalid json'
    var_20 = 'non_existent_file.json'
    var_21 = module_0.generate_context(var_20)



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


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = 'project_name'
    var_16 = 'test_project'
    var_17 = {var_15: var_16}
    var_18 = module_0.Environment()
    var_19 = var_14 / var_16
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_17}
    var_23 = True
    var_24 = 'project_name'
    var_25 = ''
    var_26 = {var_24: var_25}
    var_27 = module_0.Environment()
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'cookiecutter'
    var_30 = {var_29: var_26}
    var_31 = 'project_name'
    var_32 = 'version'
    var_33 = 'my_project'
    var_34 = '1.0'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Environment()
    var_37 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_38 = 'cookiecutter'
    var_39 = {var_38: var_35}



# Parsed testcases at query #29
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = {var_7: var_8}
    var_13 = {var_6: var_12}
    var_14 = {var_7: var_8}
    var_15 = {var_6: var_14}
    var_16 = 'hooks'
    var_17 = 'pre_gen_project.py'
    var_18 = "print('pre hook')"
    var_19 = 'post_gen_project.py'
    var_20 = "print('post hook')"
    var_21 = {var_7: var_8}
    var_22 = {var_6: var_21}
    var_23 = "raise Exception('test')"
    var_24 = 'cookiecutter'
    var_25 = 'project_name'
    var_26 = 'test_project'
    var_27 = {var_25: var_26}
    var_28 = {var_24: var_27}
    var_29 = True
    var_30 = module_0.generate_files(var_0, var_28, var_3, accept_hooks=var_29, keep_project_on_failure=var_29)
    var_31 = 'binary.bin'
    var_32 = b'\x00\x01\x02'
    var_33 = {var_7: var_8}
    var_34 = {var_6: var_33}



# Parsed testcases at query #30
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = '{{cookiecutter.project_name}}'
    var_12 = 'cookiecutter'
    var_13 = {var_12: var_9}
    var_14 = True
    var_15 = {var_12: var_9}
    var_16 = 'project_name'
    var_17 = ''
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_18}
    var_23 = 'project_name'
    var_24 = 'test'
    var_25 = {var_23: var_24}
    var_26 = module_0.Environment()
    var_27 = '{{cookiecutter.nonexistent}}'
    var_28 = 'cookiecutter'
    var_29 = {var_28: var_25}
    var_30 = 'project_name'
    var_31 = 'test_project'
    var_32 = {var_30: var_31}
    var_33 = module_0.Environment()
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'cookiecutter'
    var_36 = {var_35: var_32}
    var_37 = '{{cookiecutter.project_name}}'
    var_38 = 'cookiecutter'
    var_39 = {var_38: var_32}
    var_40 = False



# Parsed testcases at query #31
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = var_14 / var_17
    var_21 = '{{cookiecutter.project_name}}'
    var_22 = 'cookiecutter'
    var_23 = {var_22: var_18}
    var_24 = True
    var_25 = 'project_name'
    var_26 = ''
    var_27 = {var_25: var_26}
    var_28 = module_0.Environment()
    var_29 = '{{cookiecutter.project_name}}'
    var_30 = 'cookiecutter'
    var_31 = {var_30: var_27}
    var_32 = 'project_name'
    var_33 = 'author'
    var_34 = 'my_project'
    var_35 = 'test_author'
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.Environment()
    var_38 = '{{cookiecutter.project_name}}_{{cookiecutter.author}}'
    var_39 = 'cookiecutter'
    var_40 = {var_39: var_36}



# Parsed testcases at query #32
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



# Parsed testcases at query #33
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
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = 'test.txt'
    var_14 = False
    var_15 = 'binary.bin'
    var_16 = b'\x00\x01\x02\x03'
    var_17 = 'binary.bin'
    var_18 = False
    var_19 = 'test.txt'
    var_20 = True
    var_21 = 'bad.txt'
    var_22 = '{% if %}'
    var_23 = 'bad.txt'
    var_24 = False
    var_25 = []
    var_26 = {var_6: var_25, var_7: var_9}
    var_27 = {var_4: var_26}
    var_28 = 'test.txt'
    var_29 = False



# Parsed testcases at query #34
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'test'
    var_8 = 'template'
    var_9 = 'output'
    var_10 = 'cookiecutter.json'
    var_11 = '{"project_name": "test"}'
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'test'
    var_14 = True
    var_15 = 'template'
    var_16 = 'output'
    var_17 = 'cookiecutter.json'
    var_18 = '{"project_name": "test"}'
    var_19 = '{{cookiecutter.project_name}}'
    var_20 = 'file.txt'
    var_21 = 'original'
    var_22 = 'test'
    var_23 = 'modified'
    var_24 = True
    var_25 = 'template'
    var_26 = 'output'
    var_27 = 'cookiecutter.json'
    var_28 = '{"project_name": "test"}'
    var_29 = '{{cookiecutter.project_name}}'
    var_30 = 'hooks'
    var_31 = 'pre_gen_project.py'
    var_32 = var_20 / var_31
    var_33 = "import os\nwith open('hook_test.txt', 'w') as f: f.write('pre')"
    var_34 = 'post_gen_project.py'
    var_35 = "import os\nwith open('hook_test.txt', 'a') as f: f.write('post')"
    var_36 = True
    var_37 = 'test'
    assert var_37 == b'binary content'
    var_38 = 'hook_test.txt'
    var_39 = var_24 / var_38
    var_40 = 'template'
    var_41 = 'output'
    var_42 = 'cookiecutter.json'
    var_43 = '{"project_name": "default"}'
    var_44 = '{{cookiecutter.project_name}}'
    var_45 = 'file.txt'
    var_46 = var_30 / var_45
    var_47 = 'project_name'
    var_48 = 'custom'
    var_49 = {var_47: var_48}
    var_50 = var_22 / var_45
    var_51 = 'template'
    var_52 = 'output'
    var_53 = 'cookiecutter.json'
    var_54 = 'project_name'
    var_55 = '_copy_without_render'
    var_56 = 'test'
    var_57 = '*.bin'
    var_58 = [var_57]
    var_59 = {var_54: var_56, var_55: var_58}
    var_60 = module_0.dumps(var_59)
    var_61 = '{{cookiecutter.project_name}}'
    var_62 = 'file.bin'
    var_63 = var_48 / var_62
    var_64 = b'binary content'
    var_65 = 'template'
    var_66 = 'output'
    var_67 = 'cookiecutter.json'
    var_68 = '{"project_name": "test"}'
    var_69 = '{{cookiecutter.project_name}}'
    var_70 = 'file.txt'
    var_71 = var_60 / var_70
    var_72 = '{{cookiecutter.undefined_var}}'
    var_73 = 'template'
    var_74 = 'output'
    var_75 = 'cookiecutter.json'
    var_76 = '{"project_name": "test"}'
    var_77 = '{{cookiecutter.project_name}}'
    var_78 = 'test'



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mock-template/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'project_slug'
    var_7 = 'extra_slug'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = module_0.generate_context(var_0, var_4, var_8)
    var_11 = 'tests/invalid-json.json'
    var_12 = module_0.generate_context(var_11)
    var_13 = 'tests/non-existent.json'
    var_14 = module_0.generate_context(var_13)



# Parsed testcases at query #37
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
    var_8 = []
    var_9 = None
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = 'test.txt'
    var_14 = False
    var_15 = 'binary.bin'
    var_16 = b'\x00\x01\x02\x03'
    var_17 = 'binary.bin'
    var_18 = False
    var_19 = 'test.txt'
    var_20 = True
    var_21 = 'bad.txt'
    var_22 = 'Hello, {% if %}!'
    var_23 = 'bad.txt'
    var_24 = False
    var_25 = []
    var_26 = {var_6: var_25, var_7: var_9}
    var_27 = {var_4: var_26}
    var_28 = 'test.txt'
    var_29 = False



# Parsed testcases at query #38
#--------------------------


def test_case_0():
    var_0 = 'tests/mocks/valid-template'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/output'
    var_7 = True



# Parsed testcases at query #39
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mock_template/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/mock_template/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/mock_template/nonexistent.json'
    var_12 = module_0.generate_context(var_11)
    var_13 = 'tests/mock_template/empty.json'
    var_14 = module_0.generate_context(var_13)
    var_15 = []
    var_16 = 'tests/mock_template/nested.json'
    var_17 = module_0.generate_context(var_16)



# Parsed testcases at query #40
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '/tmp/test_output'
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = False
    var_8 = True
    var_9 = ''



# Parsed testcases at query #41
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



# Parsed testcases at query #42
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_context.json'
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



# Parsed testcases at query #43
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
    var_10 = 'Hello, {{ cookiecutter.name }}!'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #44
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
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



# Parsed testcases at query #45
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'test'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = 'template'
    var_13 = 'output'
    var_14 = 'cookiecutter.json'
    var_15 = '{"project_name": "test"}'
    var_16 = '{{cookiecutter.project_name}}'
    var_17 = True
    var_18 = 'test'
    var_19 = str(var_6)
    var_20 = 'template'
    var_21 = 'output'
    var_22 = 'cookiecutter.json'
    var_23 = '{"project_name": "test"}'
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'file.txt'
    var_26 = 'original'
    var_27 = 'test'
    var_28 = var_9 / var_25
    var_29 = 'modified'
    var_30 = True
    var_31 = 'template'
    var_32 = 'output'
    var_33 = 'cookiecutter.json'
    var_34 = '{"project_name": "test"}'
    var_35 = '{{cookiecutter.project_name}}'
    var_36 = 'hooks'
    var_37 = 'pre_gen_project.py'
    var_38 = var_26 / var_37
    var_39 = "print('pre hook')"
    var_40 = 'post_gen_project.py'
    var_41 = var_28 / var_40
    var_42 = "print('post hook')"
    var_43 = True
    var_44 = 'test'
    var_45 = 'template'
    var_46 = 'output'
    var_47 = 'cookiecutter.json'
    var_48 = '{"project_name": "test"}'
    var_49 = '{{cookiecutter.undefined_var}}'
    var_50 = 'template'
    var_51 = 'output'
    var_52 = 'cookiecutter'
    var_53 = 'project_name'
    var_54 = '_copy_without_render'
    var_55 = 'test'
    var_56 = '*.bin'
    var_57 = [var_56]
    var_58 = {var_53: var_55, var_54: var_57}
    var_59 = {var_52: var_58}
    var_60 = 'cookiecutter.json'
    var_61 = var_59[var_52]
    var_62 = module_0.dumps(var_61)
    var_63 = '{{cookiecutter.project_name}}'
    var_64 = 'file.bin'
    var_65 = var_28 / var_64
    var_66 = 'binary'
    var_67 = var_43 / var_64



# Parsed testcases at query #46
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project", "author": "test_author"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.author}}!'
    var_6 = 'copy_only.txt'
    var_7 = 'This file should be copied without rendering.'
    var_8 = 'copy_dir'
    var_9 = 'copy_file.txt'
    var_10 = 'This file is in a copy-only directory.'
    var_11 = 'cookiecutter'
    var_12 = 'project_name'
    var_13 = 'author'
    var_14 = '_copy_without_render'
    var_15 = 'test_project'
    var_16 = 'test_author'
    var_17 = [var_6, var_8]
    var_18 = {var_12: var_15, var_13: var_16, var_14: var_17}
    var_19 = {var_11: var_18}
    var_20 = 'output'
    var_21 = True
    var_22 = False
    var_23 = 'skip.txt'
    var_24 = 'This file should be skipped.'
    var_25 = 'existing_dir'
    var_26 = False
    var_27 = 'undefined.txt'
    var_28 = 'Hello, {{cookiecutter.undefined_variable}}!'
    var_29 = True
    var_30 = False



# Parsed testcases at query #47
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = 'test_project'
    var_7 = '*.bin'
    var_8 = [var_7]
    var_9 = '\n'
    var_10 = {var_3: var_6, var_4: var_8, var_5: var_9}
    var_11 = {var_2: var_10}
    var_12 = '.'
    var_13 = module_0.FileSystemLoader(var_12)
    var_14 = module_1.Environment(loader=var_13)
    var_15 = True
    var_16 = 'Hello, {{ cookiecutter.project_name }}!'
    var_17 = module_2.generate_file(var_0, var_1, var_11, var_14)
    var_18 = 'test.bin'
    var_19 = b'test binary content'
    var_20 = module_2.generate_file(var_0, var_18, var_11, var_14)
    var_21 = module_2.generate_file(var_0, var_1, var_11, var_14, var_15)



# Parsed testcases at query #48
#--------------------------


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = 'tests/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = '*.md'
    var_8 = 'LICENSE'
    var_9 = [var_7, var_8]
    var_10 = {var_3: var_6, var_4: var_6, var_5: var_9}
    var_11 = {var_2: var_10}
    var_12 = True
    var_13 = False
    var_14 = 'README.md'
    var_15 = 'setup.py'



# Parsed testcases at query #49
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'test'
    var_8 = 'template'
    var_9 = 'output'
    var_10 = 'cookiecutter.json'
    var_11 = '{"project_name": "test"}'
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'file.txt'
    var_14 = 'project_name'
    var_15 = 'override'
    var_16 = {var_14: var_15}
    var_17 = 'template'
    var_18 = 'existing'
    var_19 = 'cookiecutter.json'
    var_20 = '{"project_name": "test"}'
    var_21 = '{{cookiecutter.project_name}}'
    var_22 = True
    var_23 = 'test'
    var_24 = 'template'
    var_25 = 'output'
    var_26 = 'cookiecutter.json'
    var_27 = '{"project_name": "test"}'
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'file.txt'
    var_30 = 'new content'
    var_31 = 'test'
    var_32 = 'old content'
    var_33 = True
    var_34 = 'template'
    var_35 = 'output'
    var_36 = 'cookiecutter.json'
    var_37 = '{"project_name": "test"}'
    var_38 = '{{cookiecutter.project_name}}'
    var_39 = b'\x00\x01\x02\x03'
    var_40 = 'binary.bin'
    var_41 = 'test'
    var_42 = var_15 / var_40
    var_43 = 'template'
    var_44 = 'output'
    var_45 = 'project_name'
    var_46 = '_copy_without_render'
    var_47 = 'test'
    var_48 = '*.bin'
    var_49 = 'static/*'
    var_50 = [var_48, var_49]
    var_51 = {var_45: var_47, var_46: var_50}
    var_52 = 'cookiecutter.json'
    var_53 = module_0.dumps(var_51)
    var_54 = '{{cookiecutter.project_name}}'
    var_55 = 'file.bin'
    var_56 = var_31 / var_55
    var_57 = 'static'
    var_58 = 'file.txt'
    var_59 = 'template'
    var_60 = 'output'
    var_61 = 'cookiecutter.json'
    var_62 = '{"project_name": "test"}'
    var_63 = '{{cookiecutter.project_name}}'
    var_64 = 'hooks'
    var_65 = 'pre_gen_project.py'
    var_66 = "print('pre hook executed')"
    var_67 = 'post_gen_project.py'
    var_68 = "print('post hook executed')"
    var_69 = True
    var_70 = 'test'
    var_71 = False
    var_72 = 'template'
    var_73 = 'output'
    var_74 = 'cookiecutter.json'
    var_75 = '{"project_name": "test"}'
    var_76 = '{{cookiecutter.undefined_var}}'



# Parsed testcases at query #50
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = module_0.generate_context(var_0, var_4, var_7)
    var_10 = 'tests/test-data/invalid.json'
    var_11 = module_0.generate_context(var_10)
    var_12 = 'tests/test-data/nonexistent.json'
    var_13 = module_0.generate_context(var_12)



# Parsed testcases at query #51
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



# Parsed testcases at query #52
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
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
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
    var_4 = 'cookiecutter'
    var_5 = 'name'
    var_6 = '_copy_without_render'
    var_7 = '_new_lines'
    var_8 = []
    var_9 = '\n'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'World'
    var_12 = {var_4: var_10, var_5: var_11}
    var_13 = 'Existing content'
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
    var_11 = 'test.txt'
    var_12 = False



# Parsed testcases at query #53
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-fixtures/context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non_existent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'tests/test-fixtures/invalid_context.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'project_name'
    var_7 = 'default_project'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, var_8)
    var_10 = 'extra_project'
    var_11 = {var_6: var_10}
    var_12 = module_0.generate_context(var_0, extra_context=var_11)
    var_13 = module_0.generate_context(var_0, var_8, var_11)



# Parsed testcases at query #54
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'extra_project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/test_data/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/test_data/nonexistent.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #55
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/valid-template'
    var_1 = 'tests/mocks/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_6, var_1, var_7)
    var_9 = 'README.md'
    var_10 = 'setup.py'



# Parsed testcases at query #56
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'template'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = 'project_name'
    var_9 = 'overridden_project'
    var_10 = {var_8: var_9}
    var_11 = 'template'
    var_12 = 'project_name'
    var_13 = 'test_project'
    var_14 = {var_12: var_13}
    var_15 = 'template'
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = True
    var_20 = 'template'
    var_21 = 'project_name'
    var_22 = 'test_project'
    var_23 = {var_21: var_22}
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'Hello, {{cookiecutter.project_name}}!'
    var_26 = 'test.txt'
    var_27 = 'template'
    var_28 = 'project_name'
    var_29 = '_copy_without_render'
    var_30 = 'test_project'
    var_31 = '*.md'
    var_32 = [var_31]
    var_33 = {var_28: var_30, var_29: var_32}
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'This is a {{cookiecutter.project_name}} project.'
    var_36 = 'README.md'
    var_37 = 'template'
    var_38 = 'project_name'
    var_39 = 'test_project'
    var_40 = {var_38: var_39}
    var_41 = 'hooks'
    var_42 = 'import os\nos.makedirs("pre_hook_dir")'
    var_43 = True
    var_44 = 'pre_hook_dir'
    var_45 = 'template'
    var_46 = 'project_name'
    var_47 = 'test_project'
    var_48 = {var_46: var_47}
    var_49 = '{{cookiecutter.project_name}}'
    var_50 = 'Hello, {{cookiecutter.project_name}}!'
    var_51 = 'Existing content'
    var_52 = True
    var_53 = 'template'
    var_54 = 'project_name'
    var_55 = 'test_project'
    var_56 = {var_54: var_55}
    var_57 = '{{cookiecutter.project_name}}'
    var_58 = 'Hello, {{cookiecutter.project_name}}!'
    var_59 = True



# Parsed testcases at query #57
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



# Parsed testcases at query #58
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_new_lines'
    var_5 = 'test_project'
    var_6 = '\n'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ cookiecutter.project_name }}!'
    var_14 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_15 = module_3.rmtree(var_0)



# Parsed testcases at query #59
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



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = 'tests/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'author'
    var_6 = 'email'
    var_7 = 'version'
    var_8 = '_copy_without_render'
    var_9 = '_new_lines'
    var_10 = 'test_project'
    var_11 = 'Test Author'
    var_12 = 'test@example.com'
    var_13 = '0.1.0'
    var_14 = '*.bin'
    var_15 = 'static/*'
    var_16 = [var_14, var_15]
    var_17 = '\n'
    var_18 = {var_3: var_10, var_4: var_10, var_5: var_11, var_6: var_12, var_7: var_13, var_8: var_16, var_9: var_17}
    var_19 = {var_2: var_18}
    var_20 = True
    var_21 = False
    var_22 = 'README.md'
    var_23 = 'setup.py'
    var_24 = 'static'
    var_25 = 'test.bin'



# Parsed testcases at query #61
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
    var_19 = 'invalid json'
    var_20 = 'non_existent.json'
    var_21 = module_0.generate_context(var_20)



# Parsed testcases at query #62
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'nonexistent.json'
    var_4 = module_0.generate_context(var_3)
    var_5 = 'invalid json'
    var_6 = module_0.generate_context(var_5)
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = 'default'
    var_11 = {var_7: var_10}
    var_12 = 'key'
    var_13 = 'value'
    var_14 = {var_12: var_13}
    var_15 = 'extra'
    var_16 = {var_12: var_15}



# Parsed testcases at query #63
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'test_file.txt'
    var_10 = 'Hello, {{cookiecutter.author}}!'
    var_11 = 'default_project'
    var_12 = 'default_author'
    var_13 = {var_10: var_11, var_2: var_12}
    var_14 = module_0.generate_files(var_0, var_5, var_6)
    var_15 = module_1.rmtree(var_0)
    var_16 = module_1.rmtree(var_6)



# Parsed testcases at query #64
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = 'tests/templates'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = 'Hello, {{ cookiecutter.project_name }}!'
    var_11 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #65
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/basic'
    var_1 = 'tests/test-output'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'test_project'
    var_5 = 'test_slug'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True
    var_8 = 'existing content'
    assert var_8 == 'existing content'



# Parsed testcases at query #66
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



# Parsed testcases at query #67
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



# Parsed testcases at query #68
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'output'
    var_7 = True
    var_8 = 'test_project'
    var_9 = 'project_name'
    var_10 = 'override_project'
    var_11 = {var_9: var_10}
    var_12 = 'new content'
    var_13 = {var_9: var_10}
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py'
    var_16 = "\nimport os\nwith open('hook_test.txt', 'w') as f:\n    f.write('pre hook')\n"
    var_17 = 'post_gen_project.py'
    var_18 = "\nimport os\nwith open('hook_test.txt', 'a') as f:\n    f.write('post hook')\n"
    var_19 = 'hook_project'
    var_20 = {var_9: var_19}
    var_21 = 'hook_test.txt'
    var_22 = 'bad_template'
    var_23 = '{{cookiecutter.undefined_var}}'
    var_24 = 'project_name'
    var_25 = 'test'
    var_26 = {var_24: var_25}
    var_27 = True



# Parsed testcases at query #69
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'extra_project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/test_data/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/test_data/nonexistent.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #70
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test.txt'
    var_2 = 'Hello {{ name }}!'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'World'
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = module_0.Environment()
    var_13 = False
    var_14 = 'binary.bin'
    var_15 = b'test binary content'
    var_16 = True



# Parsed testcases at query #71
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



# Parsed testcases at query #72
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/basic-template'
    var_1 = 'tests/test-output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_3: var_6, var_4: var_6, var_5: var_7}
    var_9 = {var_2: var_8}
    var_10 = True
    var_11 = 'README.md'
    var_12 = 'setup.py'



# Parsed testcases at query #73
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
    var_11 = 'project_name'
    var_12 = 'test_project'
    var_13 = {var_11: var_12}
    var_14 = module_0.Environment()
    var_15 = '{{ cookiecutter.project_name }}'
    var_16 = 'test.txt'
    var_17 = 'original content'
    var_18 = True
    var_19 = 'project_name'
    var_20 = ''
    var_21 = {var_19: var_20}
    var_22 = module_0.Environment()
    var_23 = '{{ cookiecutter.project_name }}'
    var_24 = 'project_name'
    var_25 = 'test_project'
    var_26 = {var_24: var_25}
    var_27 = module_0.Environment()
    var_28 = '{{ cookiecutter.nonexistent_var }}'



# Parsed testcases at query #74
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = '.'
    var_8 = module_0.FileSystemLoader(var_7)
    var_9 = module_1.Environment(loader=var_8)
    var_10 = var_6[var_2]
    var_11 = {var_2: var_10}
    var_12 = 'Hello, {{ cookiecutter.project_name }}!'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9)



# Parsed testcases at query #75
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'tests/test-templates/basic'
    var_1 = 'tests/output/basic'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = True
    var_6 = module_0.generate_files(var_0, var_4, var_1, var_5)
    var_7 = module_0.generate_files(var_0, var_4, var_1, var_5, var_5)
    var_8 = 'tests/test-templates/with-hooks'
    var_9 = 'tests/output/with-hooks'
    var_10 = module_0.generate_files(var_8, var_4, var_9, var_5, accept_hooks=var_5)
    var_11 = 'tests/test-templates/undefined-variable'
    var_12 = 'tests/output/undefined-variable'
    var_13 = True
    var_14 = module_0.generate_files(var_11, var_4, var_12, var_13)
    var_15 = 'tests/output/exists'
    var_16 = False
    var_17 = module_0.generate_files(var_0, var_4, var_15, var_16)
    var_18 = module_1.rmtree(var_1)
    var_19 = module_1.rmtree(var_9)
    var_20 = module_1.rmtree(var_12)
    var_21 = module_1.rmtree(var_15)



# Parsed testcases at query #76
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/simple'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/output'
    var_7 = True
    var_8 = False
    var_9 = False
    var_10 = False
    var_11 = 'README.md'
    var_12 = 'setup.py'



# Parsed testcases at query #77
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



# Parsed testcases at query #78
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
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



# Parsed testcases at query #79
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = '{{cookiecutter.project_name}}'
    var_12 = 'cookiecutter'
    var_13 = {var_12: var_9}
    var_14 = True
    var_15 = {var_12: var_9}
    var_16 = 'project_name'
    var_17 = ''
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_18}
    var_23 = 'project_name'
    var_24 = 'test_project'
    var_25 = {var_23: var_24}
    var_26 = module_0.Environment()
    var_27 = '{{cookiecutter.project_name}}'
    var_28 = 'cookiecutter'
    var_29 = {var_28: var_25}
    var_30 = '{{cookiecutter.project_name}}'
    var_31 = 'cookiecutter'
    var_32 = {var_31: var_25}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/data/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non_existent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'tests/data/invalid.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'project_name'
    var_7 = 'default_project'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, var_8)
    var_10 = 'extra_project'
    var_11 = {var_6: var_10}
    var_12 = module_0.generate_context(var_0, extra_context=var_11)
    var_13 = module_0.generate_context(var_0, var_8, var_11)



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'my_{{ cookiecutter.project_name }}_project'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = '/tmp/test_output'
    var_7 = module_0.Environment()
    var_8 = 'existing'
    var_9 = True
    var_10 = 'existing'
    var_11 = ''
    var_12 = 'my_{{ cookiecutter.nonexistent }}_project'



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
    var_26 = [var_13, var_14]
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
    var_43 = {var_34: var_5, var_35: var_9}
    var_44 = {var_33: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_42, var_44)
    var_46 = True
    var_47 = {var_33: var_46}
    var_48 = 'yes'
    var_49 = {var_33: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_33: var_46}
    var_52 = 'invalid_bool'
    var_53 = {var_33: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = 'nested'
    var_56 = {var_34: var_2}
    var_57 = {var_55: var_56}
    var_58 = {var_54: var_57}
    var_59 = {var_34: var_5}
    var_60 = {var_55: var_59}
    var_61 = {var_54: var_60}
    var_62 = module_0.apply_overwrites_to_context(var_58, var_61)



# Parsed testcases at query #4
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
    var_20 = ''
    var_21 = module_0.is_copy_only_path(var_20, var_6)
    assert var_21 is False



# Parsed testcases at query #5
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = module_0.generate_context()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'invalid json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'key1'
    var_7 = 'default_value'
    var_8 = {var_6: var_7}
    var_9 = 'key2'
    var_10 = 'extra_value'
    var_11 = 'extra_value2'
    var_12 = {var_6: var_10, var_9: var_11}
    var_13 = module_0.generate_context(default_context=var_8, extra_context=var_12)
    var_14 = 'w'
    var_15 = '.json'
    var_16 = False
    var_17 = 'choices'
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 'c'
    var_21 = [var_18, var_19, var_20]
    var_22 = {var_17: var_21}
    var_23 = [var_18, var_20]
    var_24 = {var_17: var_23}
    var_25 = [var_18, var_19, var_20]
    var_26 = {var_17: var_25}
    var_27 = 'd'
    var_28 = [var_18, var_27]
    var_29 = {var_17: var_28}
    var_30 = module_0.generate_context(var_4, extra_context=var_29)
    var_31 = 'choice'
    var_32 = [var_18, var_19, var_20]
    var_33 = {var_31: var_32}
    var_34 = {var_31: var_19}
    var_35 = [var_18, var_19, var_20]
    var_36 = {var_31: var_35}
    var_37 = {var_31: var_27}
    var_38 = module_0.generate_context(var_4, extra_context=var_37)
    var_39 = 'bool_var'
    var_40 = True
    var_41 = {var_39: var_40}
    var_42 = 'yes'
    var_43 = {var_39: var_42}
    var_44 = {var_39: var_40}
    var_45 = 'invalid'
    var_46 = {var_39: var_45}
    var_47 = module_0.generate_context(var_4, extra_context=var_46)
    var_48 = 'dict_var'
    var_49 = 'value1'
    var_50 = 'value2'
    var_51 = {var_47: var_49, var_9: var_50}
    var_52 = {var_48: var_51}
    var_53 = 'new_value1'
    var_54 = {var_47: var_53}
    var_55 = {var_48: var_54}



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
    var_6 = '{{ test }}_dir'
    var_7 = True
    var_8 = False
    var_9 = ''
    var_10 = 'test_dir'
    var_11 = 'value_dir'



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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
    var_12 = 'choice_var'
    var_13 = 'option1'
    var_14 = 'option2'
    var_15 = 'option3'
    var_16 = [var_13, var_14, var_15]
    var_17 = {var_12: var_16}
    var_18 = {var_12: var_14}
    var_19 = module_0.apply_overwrites_to_context(var_17, var_18)
    var_20 = [var_13, var_14]
    var_21 = {var_12: var_20}
    var_22 = 'invalid_option'
    var_23 = {var_12: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = 'multi_var'
    var_26 = [var_13, var_14, var_15]
    var_27 = {var_25: var_26}
    var_28 = [var_13, var_15]
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
    var_44 = 'no'
    var_45 = {var_36: var_44}
    var_46 = module_0.apply_overwrites_to_context(var_43, var_45)
    var_47 = {var_36: var_37}
    var_48 = 'invalid_bool'
    var_49 = {var_36: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = 'nested'
    var_52 = {var_50: var_2, var_1: var_3}
    var_53 = {var_51: var_52}
    var_54 = {var_50: var_5}
    var_55 = {var_51: var_54}
    var_56 = module_0.apply_overwrites_to_context(var_53, var_55)
    var_57 = {var_50: var_2}
    var_58 = {var_51: var_57}
    var_59 = {var_50: var_5, var_1: var_9}
    var_60 = {var_51: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_58, var_60)



# Parsed testcases at query #10
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'cookiecutter.json'
    var_7 = 'project_name'
    var_8 = 'author'
    var_9 = 'test_project'
    var_10 = 'Test Author'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'project_name'
    var_13 = 'default_project'
    var_14 = {var_12: var_13}
    var_15 = 'cookiecutter.json'
    var_16 = 'project_name'
    var_17 = 'author'
    var_18 = 'test_project'
    var_19 = 'Test Author'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'project_name'
    var_22 = 'extra_project'
    var_23 = {var_21: var_22}
    var_24 = 'cookiecutter.json'
    var_25 = '{invalid json}'
    var_26 = 'non_existent_file.json'
    var_27 = module_0.generate_context(var_26)



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'Hello'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"name": "default"}'
    var_13 = '{{cookiecutter.name}}.txt'
    var_14 = 'Content'
    var_15 = 'name'
    var_16 = 'custom'
    var_17 = {var_15: var_16}
    var_18 = True
    var_19 = 'custom.txt'
    var_20 = 'template'
    var_21 = 'output'
    var_22 = 'cookiecutter.json'
    var_23 = '{"name": "test"}'
    var_24 = '{{cookiecutter.name}}.txt'
    var_25 = 'Original'
    var_26 = True
    var_27 = 'Modified'
    var_28 = 'test.txt'
    var_29 = 'template'
    var_30 = 'output'
    var_31 = 'cookiecutter.json'
    var_32 = '{"name": "test"}'
    var_33 = 'hooks'
    var_34 = 'pre_gen_project.py'
    var_35 = "import os\nwith open('pre_hook.txt', 'w') as f: f.write('pre')"
    var_36 = 'post_gen_project.py'
    var_37 = var_27 / var_36
    assert var_37 == b'\x00\x01\x02'
    var_38 = "import os\nwith open('post_hook.txt', 'w') as f: f.write('post')"
    var_39 = True
    var_40 = 'pre_hook.txt'
    var_41 = 'post_hook.txt'
    var_42 = 'template'
    var_43 = 'output'
    var_44 = 'cookiecutter.json'
    var_45 = '{"name": "test"}'
    var_46 = '{{cookiecutter.undefined}}.txt'
    var_47 = 'Content'
    var_48 = True
    var_49 = 'template'
    var_50 = 'output'
    var_51 = 'cookiecutter.json'
    var_52 = '{"name": "test", "_copy_without_render": ["*.bin"]}'
    var_53 = '{{cookiecutter.name}}.bin'
    var_54 = b'\x00\x01\x02'
    var_55 = True
    var_56 = 'test.bin'



# Parsed testcases at query #12
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = '{{cookiecutter.project_name}}'
    var_12 = 'cookiecutter'
    var_13 = {var_12: var_9}
    var_14 = True
    var_15 = {var_12: var_9}
    var_16 = 'project_name'
    var_17 = ''
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_18}
    var_23 = 'project_name'
    var_24 = 'test_project'
    var_25 = {var_23: var_24}
    var_26 = module_0.Environment()
    var_27 = '{{cookiecutter.project_name}}'
    var_28 = 'cookiecutter'
    var_29 = {var_28: var_25}
    var_30 = '{{cookiecutter.project_name}}'
    var_31 = 'cookiecutter'
    var_32 = {var_31: var_25}



# Parsed testcases at query #13
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
    var_10 = '/tmp/test_template'
    var_11 = True
    var_12 = 'Hello, {{ cookiecutter.name }}!'
    var_13 = module_2.generate_file(var_0, var_1, var_6, var_9)
    var_14 = 'test_binary.bin'
    var_15 = b'\x00\x01\x02\x03'
    var_16 = module_2.generate_file(var_0, var_14, var_6, var_9)
    var_17 = module_2.generate_file(var_0, var_1, var_6, var_9, var_11)
    var_18 = module_3.rmtree(var_0)
    var_19 = module_3.rmtree(var_10)



# Parsed testcases at query #14
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mock_project_template/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'Extra Project'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/mock_project_template/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/mock_project_template/nonexistent.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #15
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



# Parsed testcases at query #16
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'template.txt'
    var_3 = 'name'
    var_4 = 'World'
    var_5 = {var_3: var_4}
    var_6 = module_0.Environment()
    var_7 = 'cookiecutter'
    var_8 = {var_7: var_5}
    var_9 = 'binary.bin'
    var_10 = b'test binary content'
    var_11 = {var_7: var_5}
    var_12 = {var_7: var_5}
    var_13 = True
    var_14 = 'Line 1\nLine 2\r\nLine 3'
    var_15 = 'newlines.txt'
    var_16 = {var_7: var_5}



# Parsed testcases at query #17
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1

def test_case_0():
    var_0 = 'project'
    var_1 = True
    var_2 = 'test_file.txt'
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
    var_13 = '.'
    var_14 = module_0.FileSystemLoader(var_13)
    var_15 = module_1.Environment(loader=var_14)
    var_16 = 'test_binary.bin'
    var_17 = b'binary content'
    var_18 = 'bad_template.txt'
    var_19 = '{% if %}'
    var_20 = []
    var_21 = {var_6: var_20, var_7: var_9}
    var_22 = {var_4: var_21}



# Parsed testcases at query #18
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
    var_10 = 'another_project'
    var_11 = {var_9: var_10}
    var_12 = 'Existing content'
    var_13 = 'hooks'
    var_14 = 'pre_gen_project.py'
    var_15 = "\nimport os\nwith open(os.path.join(project_dir, 'hook_test.txt'), 'w') as f:\n    f.write('pre hook executed')\n"
    var_16 = 'hook_test.txt'
    var_17 = '{"project_name": "{{undefined_var}}"}'
    var_18 = True



# Parsed testcases at query #19
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



# Parsed testcases at query #20
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'project_name'
    var_6 = 'test_project'
    var_7 = {var_5: var_6}
    var_8 = module_0.Environment()
    var_9 = var_4 / var_6
    var_10 = '{{cookiecutter.project_name}}'
    var_11 = False
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = True
    var_14 = 'project_name'
    var_15 = ''
    var_16 = {var_14: var_15}
    var_17 = module_0.Environment()
    var_18 = '{{cookiecutter.project_name}}'
    var_19 = 'project_name'
    var_20 = 'test'
    var_21 = {var_19: var_20}
    var_22 = module_0.Environment()
    var_23 = '{{cookiecutter.nonexistent}}'



# Parsed testcases at query #21
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
    var_43 = 'test content {{ undefined_var }}'
    var_44 = 'output'
    var_45 = True
    var_46 = 'test'



# Parsed testcases at query #22
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = module_0.Environment()
    var_5 = False
    var_6 = True
    var_7 = ''



# Parsed testcases at query #23
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = True
    var_8 = ''



# Parsed testcases at query #24
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/simple'
    var_1 = 'tests/test-output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = '*.txt'
    var_8 = [var_7]
    var_9 = {var_3: var_6, var_4: var_6, var_5: var_8}
    var_10 = {var_2: var_9}
    var_11 = True
    var_12 = False
    var_13 = 'README.md'
    var_14 = 'test.txt'



# Parsed testcases at query #25
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"project_name": "test"}'
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'cookiecutter'
    var_15 = 'project_name'
    var_16 = 'override'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = 'template'
    var_20 = 'output'
    var_21 = 'cookiecutter.json'
    var_22 = '{"project_name": "test"}'
    var_23 = '{{cookiecutter.project_name}}'
    var_24 = 'file.txt'
    var_25 = 'old'
    var_26 = True
    var_27 = 'new'
    var_28 = 'test'
    var_29 = 'template'
    var_30 = 'output'
    var_31 = 'cookiecutter.json'
    var_32 = '{"project_name": "test"}'
    var_33 = '{{cookiecutter.project_name}}'
    var_34 = 'hooks'
    var_35 = 'pre_gen_project.py'
    var_36 = var_24 / var_35
    var_37 = "\nimport os\nwith open(os.path.join('{{cookiecutter.project_name}}', 'hook.txt'), 'w') as f:\n    f.write('pre')\n"
    var_38 = True
    var_39 = 'test'
    var_40 = 'hook.txt'
    var_41 = 'template'
    var_42 = 'output'
    var_43 = 'cookiecutter.json'
    var_44 = '{"project_name": "test"}'
    var_45 = '{{cookiecutter.project_name}}'
    var_46 = b'\x00\x01\x02\x03'
    var_47 = 'binary.bin'
    var_48 = var_34 / var_47
    var_49 = 'test'
    var_50 = var_26 / var_47
    var_51 = 'template'
    var_52 = 'output'
    var_53 = 'cookiecutter.json'
    var_54 = '{"project_name": "test"}'
    var_55 = '{{cookiecutter.undefined_var}}'



# Parsed testcases at query #26
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test_cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.generate_context(var_0)
    var_7 = 'project_name'
    var_8 = 'default_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.generate_context(var_0, var_9)
    var_11 = 'author'
    var_12 = 'extra_author'
    var_13 = {var_11: var_12}
    var_14 = None
    var_15 = module_0.generate_context(var_0, var_14, var_13)
    var_16 = 'invalid json'
    var_17 = module_0.generate_context(var_0)



# Parsed testcases at query #27
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = {var_7: var_8}
    var_13 = {var_6: var_12}
    var_14 = 'hooks'
    var_15 = 'pre_gen_project.py'
    var_16 = "print('pre hook')"
    var_17 = 'post_gen_project.py'
    var_18 = "print('post hook')"
    var_19 = {var_7: var_8}
    var_20 = {var_6: var_19}
    var_21 = 'binary.bin'
    var_22 = b'\x00\x01\x02\x03'
    var_23 = '{"project_name": "test_project", "_copy_without_render": ["*.bin"]}'
    var_24 = {var_7: var_8}
    var_25 = {var_6: var_24}
    var_26 = '{{cookiecutter.undefined_var}}'
    var_27 = 'cookiecutter'
    var_28 = 'project_name'
    var_29 = 'test_project'
    var_30 = {var_28: var_29}
    var_31 = {var_27: var_30}
    var_32 = True
    var_33 = module_0.generate_files(var_0, var_31, var_3, var_32)



# Parsed testcases at query #28
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
    var_6 = None
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_2: var_7}
    var_9 = '.'
    var_10 = module_0.FileSystemLoader(var_9)
    var_11 = module_1.Environment(loader=var_10)
    var_12 = True
    var_13 = 'Hello, {{ name }}!'
    var_14 = 'test.bin'
    var_15 = b'test binary content'
    assert var_15 == 'Hello, World!'
    assert var_15 == b'test binary content'
    var_16 = module_2.generate_file(var_0, var_1, var_8, var_11)
    var_17 = module_2.generate_file(var_0, var_14, var_8, var_11)
    var_18 = module_2.generate_file(var_0, var_1, var_8, var_11, var_12)
    var_19 = ''
    var_20 = module_2.generate_file(var_0, var_19, var_8, var_11)
    var_21 = []
    var_22 = {var_20: var_21, var_4: var_6}
    var_23 = {var_19: var_22}
    var_24 = module_2.generate_file(var_0, var_1, var_23, var_11)
    var_25 = module_3.rmtree(var_0)



# Parsed testcases at query #29
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'default_project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'author'
    var_7 = 'extra_author'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = module_0.generate_context(var_0, var_4, var_8)
    var_11 = 'tests/mocks/invalid.json'
    var_12 = module_0.generate_context(var_11)
    var_13 = 'tests/mocks/nonexistent.json'
    var_14 = module_0.generate_context(var_13)



# Parsed testcases at query #30
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
    var_9 = 'test_dir'
    var_10 = {var_1: var_2}
    var_11 = True
    var_12 = False
    var_13 = ''



# Parsed testcases at query #31
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"name": "default"}'
    var_13 = '{{cookiecutter.name}}'
    var_14 = 'file.txt'
    var_15 = 'name'
    var_16 = 'custom'
    var_17 = {var_15: var_16}
    var_18 = True
    var_19 = 'template'
    var_20 = 'output'
    var_21 = 'cookiecutter.json'
    var_22 = '{"name": "test"}'
    var_23 = '{{cookiecutter.name}}'
    var_24 = 'file.txt'
    var_25 = 'original'
    var_26 = True
    var_27 = var_18 / var_24
    var_28 = 'modified'
    var_29 = 'test'
    var_30 = 'template'
    var_31 = 'output'
    var_32 = 'cookiecutter.json'
    var_33 = '{"name": "test"}'
    var_34 = '{{cookiecutter.name}}'
    var_35 = 'hooks'
    var_36 = 'pre_gen_project.py'
    var_37 = var_24 / var_36
    var_38 = "open('hook_marker', 'w').write('pre')"
    var_39 = 'post_gen_project.py'
    var_40 = var_26 / var_39
    var_41 = "open('hook_marker', 'a').write('post')"
    var_42 = True
    var_43 = 'hook_marker'
    var_44 = 'template'
    var_45 = 'output'
    var_46 = 'cookiecutter.json'
    var_47 = '{"name": "test"}'
    var_48 = '{{cookiecutter.undefined_var}}'
    var_49 = True



# Parsed testcases at query #32
#--------------------------


def test_case_0():
    var_0 = 'tests/test-template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'tests/output'
    var_7 = True
    var_8 = False
    var_9 = 'README.md'
    var_10 = 'src'



# Parsed testcases at query #33
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'test.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'nonexistent.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'invalid.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'test.json'
    var_7 = 'key'
    var_8 = 'new_value'
    var_9 = {var_7: var_8}
    var_10 = module_0.generate_context(var_6, var_9)
    var_11 = 'test.json'
    var_12 = 'key'
    var_13 = 'extra_value'
    var_14 = {var_12: var_13}
    var_15 = module_0.generate_context(var_11, extra_context=var_14)
    var_16 = 'test.json'
    var_17 = 'key'
    var_18 = 'default_value'
    var_19 = {var_17: var_18}
    var_20 = 'extra_value'
    var_21 = {var_17: var_20}
    var_22 = module_0.generate_context(var_16, var_19, var_21)



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
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = 'project_name'
    var_9 = 'author'
    var_10 = 'test_project'
    var_11 = 'test_author'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'Hello, {{ cookiecutter.project_name }}!'
    assert var_13 == 'Hello, test_project!'
    var_14 = module_0.generate_files(var_0, var_5, var_6)
    var_15 = 'test_file.txt'
    var_16 = module_1.rmtree(var_0)
    var_17 = module_1.rmtree(var_6)



# Parsed testcases at query #36
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = var_14 / var_17
    var_21 = 'test.txt'
    var_22 = var_20 / var_21
    var_23 = 'old content'
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'cookiecutter'
    var_26 = {var_25: var_18}
    var_27 = True
    var_28 = 'project_name'
    var_29 = ''
    var_30 = {var_28: var_29}
    var_31 = module_0.Environment()
    var_32 = '{{cookiecutter.project_name}}'
    var_33 = 'cookiecutter'
    var_34 = {var_33: var_30}
    var_35 = 'project_name'
    var_36 = 'version'
    var_37 = 'my_project'
    var_38 = '1.0'
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = module_0.Environment()
    var_41 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_42 = 'cookiecutter'
    var_43 = {var_42: var_39}



# Parsed testcases at query #37
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter.json'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'test_author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'cookiecutter.json'
    var_7 = 'project_name'
    var_8 = 'author'
    var_9 = 'test_project'
    var_10 = 'test_author'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'project_name'
    var_13 = 'default_project'
    var_14 = {var_12: var_13}
    var_15 = 'cookiecutter.json'
    var_16 = 'project_name'
    var_17 = 'author'
    var_18 = 'test_project'
    var_19 = 'test_author'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'project_name'
    var_22 = 'extra_project'
    var_23 = {var_21: var_22}
    var_24 = 'cookiecutter.json'
    var_25 = 'invalid json'
    var_26 = 'non_existent_file.json'
    var_27 = module_0.generate_context(var_26)



# Parsed testcases at query #38
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = 'test_template.txt'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = '_copy_without_render'
    var_5 = '_new_lines'
    var_6 = []
    var_7 = '\n'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'test_project'
    var_10 = {var_2: var_8, var_3: var_9}
    var_11 = '.'
    var_12 = module_0.FileSystemLoader(var_11)
    var_13 = module_1.Environment(loader=var_12)
    var_14 = 'Hello, {{ project_name }}!'
    assert var_14 == 'Hello, test_project!'
    var_15 = module_2.generate_file(var_0, var_1, var_10, var_13)
    var_16 = 'test_binary.bin'
    var_17 = b'binary data'
    assert var_17 == b'binary data'
    assert var_17 == 'Hello, test_project!'
    var_18 = module_2.generate_file(var_0, var_16, var_10, var_13)
    var_19 = True
    var_20 = module_2.generate_file(var_0, var_1, var_10, var_13, var_19)
    var_21 = module_3.rmtree(var_0)



# Parsed testcases at query #39
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = 'test'
    var_8 = 'template'
    var_9 = 'output'
    var_10 = 'cookiecutter.json'
    var_11 = '{"project_name": "test"}'
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'test'
    var_14 = True
    var_15 = 'template'
    var_16 = 'output'
    var_17 = 'cookiecutter.json'
    var_18 = '{"project_name": "test"}'
    var_19 = '{{cookiecutter.project_name}}'
    var_20 = 'file.txt'
    var_21 = 'content'
    var_22 = 'test'
    var_23 = 0.1
    var_24 = True
    var_25 = 'template'
    var_26 = 'output'
    var_27 = 'cookiecutter.json'
    var_28 = '{"project_name": "default"}'
    var_29 = '{{cookiecutter.project_name}}'
    var_30 = 'file.txt'
    var_31 = 'project_name'
    var_32 = 'custom'
    var_33 = {var_31: var_32}
    var_34 = 'template'
    var_35 = 'output'
    var_36 = 'cookiecutter.json'
    var_37 = '{"project_name": "test"}'
    var_38 = '{{cookiecutter.project_name}}'
    var_39 = 'hooks'
    var_40 = 'pre_gen_project.py'
    var_41 = var_30 / var_40
    var_42 = "print('pre hook')"
    var_43 = 'post_gen_project.py'
    var_44 = "print('post hook')"
    var_45 = True
    var_46 = 'test'
    var_47 = 'template'
    var_48 = 'output'
    var_49 = 'cookiecutter.json'
    var_50 = '{"project_name": "test"}'
    var_51 = '{{cookiecutter.project_name}}'
    var_52 = b'\x00\x01\x02\x03'
    var_53 = 'binary.bin'
    var_54 = var_39 / var_53
    var_55 = module_0.generate_files(var_42, output_dir=var_32)
    var_56 = 'test'
    var_57 = var_22 / var_53
    var_58 = 'template'
    var_59 = 'output'
    var_60 = 'cookiecutter.json'
    var_61 = '{"project_name": "test", "_copy_without_render": ["*.txt"]}'
    var_62 = '{{cookiecutter.project_name}}'
    var_63 = 'file.txt'
    var_64 = var_39 / var_63
    var_65 = module_0.generate_files(var_42, output_dir=var_32)
    var_66 = 'test'
    var_67 = var_22 / var_63
    var_68 = 'template'
    var_69 = 'output'
    var_70 = 'cookiecutter.json'
    var_71 = '{"project_name": "test"}'
    var_72 = '{{cookiecutter.project_name}}'
    var_73 = 'file.txt'
    var_74 = var_39 / var_73
    var_75 = '{{cookiecutter.undefined_var}}'
    var_76 = 'template'
    var_77 = 'output'
    var_78 = 'cookiecutter.json'
    var_79 = '{"project_name": "test"}'
    var_80 = '{{cookiecutter.project_name}}'
    var_81 = 'file.txt'
    var_82 = var_39 / var_81
    var_83 = '{{cookiecutter.undefined_var}}'
    var_84 = True
    var_85 = 'test'



# Parsed testcases at query #40
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/{{cookiecutter.project_name}}'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = '_copy_without_render'
    var_5 = 'test_project'
    var_6 = '*.md'
    var_7 = [var_6]
    var_8 = {var_2: var_5, var_3: var_5, var_4: var_7}
    var_9 = {var_1: var_8}
    var_10 = 'tests/output'
    var_11 = True
    var_12 = False
    var_13 = 'README.md'
    var_14 = 'src'



# Parsed testcases at query #41
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = 'test_project'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'cookiecutter'
    var_17 = {var_16: var_9}
    var_18 = True
    var_19 = 'project_name'
    var_20 = ''
    var_21 = {var_19: var_20}
    var_22 = module_0.Environment()
    var_23 = '{{cookiecutter.project_name}}'
    var_24 = 'cookiecutter'
    var_25 = {var_24: var_21}
    var_26 = 'project_name'
    var_27 = 'version'
    var_28 = 'my_project'
    var_29 = '1.0'
    var_30 = {var_26: var_28, var_27: var_29}
    var_31 = module_0.Environment()
    var_32 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_33 = 'cookiecutter'
    var_34 = {var_33: var_30}



# Parsed testcases at query #42
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'test_project'
    var_7 = True
    var_8 = 'project_name'
    var_9 = 'custom_project'
    var_10 = {var_8: var_9}
    var_11 = False



# Parsed testcases at query #43
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = '/tmp'
    var_4 = module_0.Environment()
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = '{{cookiecutter.project_name}}'
    var_7 = True
    var_8 = ''
    var_9 = 'version'
    var_10 = 'my_project'
    var_11 = '1.0'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_14 = 'my_project-1.0'



# Parsed testcases at query #44
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'existing'
    var_8 = var_0 / var_7
    var_9 = 'project_name'
    var_10 = {var_9: var_7}
    var_11 = module_0.Environment()
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_10}
    var_15 = 'existing'
    var_16 = var_12 / var_15
    var_17 = 'project_name'
    var_18 = {var_17: var_15}
    var_19 = module_0.Environment()
    var_20 = '{{cookiecutter.project_name}}'
    var_21 = 'cookiecutter'
    var_22 = {var_21: var_18}
    var_23 = True
    var_24 = 'project_name'
    var_25 = ''
    var_26 = {var_24: var_25}
    var_27 = module_0.Environment()
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'cookiecutter'
    var_30 = {var_29: var_26}
    var_31 = 'project_name'
    var_32 = 'version'
    var_33 = 'my_project'
    var_34 = '1.0'
    var_35 = {var_31: var_33, var_32: var_34}
    var_36 = module_0.Environment()
    var_37 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_38 = 'cookiecutter'
    var_39 = {var_38: var_35}



# Parsed testcases at query #45
#--------------------------


def test_case_0():
    var_0 = 'tests/mocks/basic_template'
    assert var_0 == 'This file should not be rendered'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'tests/output'
    var_7 = True
    var_8 = 'skip_file'
    var_9 = 'skip_me'
    var_10 = {var_2: var_3, var_8: var_9}
    var_11 = {var_1: var_10}
    var_12 = 'tests/mocks/template_with_hooks'
    var_13 = {var_2: var_3}
    var_14 = {var_1: var_13}
    var_15 = 'hook_output.txt'
    var_16 = 'tests/mocks/template_with_error'
    var_17 = {var_2: var_3}
    var_18 = {var_1: var_17}
    var_19 = 'tests/mocks/template_with_copy_only'
    var_20 = '_copy_without_render'
    var_21 = '*.txt'
    var_22 = [var_21]
    var_23 = {var_2: var_3, var_20: var_22}
    var_24 = {var_1: var_23}
    var_25 = 'copy_only.txt'



# Parsed testcases at query #46
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'non_existent_file.json'
    var_3 = module_0.generate_context(var_2)
    var_4 = 'tests/test-data/invalid_context.json'
    var_5 = module_0.generate_context(var_4)
    var_6 = 'project_name'
    var_7 = 'Default Project'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, var_8)
    var_10 = 'Extra Project'
    var_11 = {var_6: var_10}
    var_12 = module_0.generate_context(var_0, extra_context=var_11)
    var_13 = module_0.generate_context(var_0, var_8, var_11)



# Parsed testcases at query #47
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_5 == 'Hello, test_project!'
    var_6 = 'output'
    var_7 = 'test_project'
    var_8 = 'test.txt'
    var_9 = 'template'
    var_10 = 'project_name'
    var_11 = 'test_project'
    var_12 = {var_10: var_11}
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_14 == 'Hello, custom_project!'
    var_15 = 'output'
    var_16 = 'project_name'
    var_17 = 'custom_project'
    var_18 = {var_16: var_17}
    var_19 = 'test.txt'
    var_20 = 'template'
    var_21 = 'project_name'
    var_22 = 'test_project'
    var_23 = {var_21: var_22}
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'Hello, {{cookiecutter.project_name}}!'
    var_26 = 'output'
    var_27 = 'test_project'
    var_28 = 'test.txt'
    var_29 = False
    var_30 = True
    var_31 = 'template'
    var_32 = 'project_name'
    var_33 = 'test_project'
    var_34 = {var_32: var_33}
    var_35 = '{{cookiecutter.project_name}}'
    var_36 = 'Hello, {{cookiecutter.project_name}}!'
    var_37 = 'output'
    var_38 = 'test_project'
    var_39 = 'test.txt'
    var_40 = 'Modified content'
    assert var_40 == 'Modified content'
    var_41 = True
    var_42 = 'template'
    var_43 = 'project_name'
    var_44 = 'test_project'
    var_45 = {var_43: var_44}
    var_46 = 'hooks'
    var_47 = 'print("Pre hook executed")'
    var_48 = 'print("Post hook executed")'
    var_49 = '{{cookiecutter.project_name}}'
    var_50 = 'Hello, {{cookiecutter.project_name}}!'
    var_51 = 'output'
    var_52 = True
    var_53 = 'test_project'
    var_54 = 'test.txt'
    var_55 = 'template'
    var_56 = 'project_name'
    var_57 = 'test_project'
    var_58 = {var_56: var_57}
    var_59 = '{{cookiecutter.project_name}}'
    var_60 = 'Hello, {{cookiecutter.undefined_var}}!'
    var_61 = 'output'



# Parsed testcases at query #48
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
    var_10 = True
    var_11 = 'project_name'
    var_12 = ''
    var_13 = {var_11: var_12}
    var_14 = module_0.Environment()
    var_15 = '{{ cookiecutter.project_name }}'
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = '{{ cookiecutter.nonexistent_variable }}'



# Parsed testcases at query #49
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test-data/context.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'default'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'extra'
    var_7 = {var_2: var_6}
    var_8 = module_0.generate_context(var_0, extra_context=var_7)
    var_9 = 'tests/test-data/invalid.json'
    var_10 = module_0.generate_context(var_9)
    var_11 = 'tests/test-data/nonexistent.json'
    var_12 = module_0.generate_context(var_11)



# Parsed testcases at query #50
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'Hello'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = 'generated'
    var_10 = {var_8: var_9}
    var_11 = {var_7: var_10}
    var_12 = {var_8: var_9}
    var_13 = {var_7: var_12}
    var_14 = True
    var_15 = {var_8: var_9}
    var_16 = {var_7: var_15}
    var_17 = 'hooks'
    var_18 = 'pre_gen_project.py'
    var_19 = "import os\nos.makedirs('hook_dir')"
    var_20 = 'with_hooks'
    var_21 = {var_8: var_20}
    var_22 = {var_7: var_21}
    var_23 = 'hook_dir'
    var_24 = 'cookiecutter'
    var_25 = {}
    var_26 = {var_24: var_25}



# Parsed testcases at query #51
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello {{cookiecutter.project_name}}'
    var_6 = 'cookiecutter'
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = {var_6: var_9}
    var_11 = True
    var_12 = {var_7: var_8}
    var_13 = {var_6: var_12}
    var_14 = 'cookiecutter'
    var_15 = 'project_name'
    var_16 = 'test_project'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = False
    var_20 = 'cookiecutter'
    var_21 = 'project_name'
    var_22 = 'test_project'
    var_23 = {var_21: var_22}
    var_24 = {var_20: var_23}
    var_25 = 'new'
    var_26 = True
    var_27 = 'binary.bin'
    var_28 = b'\x00\x01\x02\x03'
    var_29 = '{"project_name": "test_project", "_copy_without_render": ["*.bin"]}'
    var_30 = {var_7: var_8}
    var_31 = {var_6: var_30}
    var_32 = 'binary_test'



# Parsed testcases at query #52
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'test.txt'
    var_5 = 'Hello, {{cookiecutter.project_name}}!'
    var_6 = 'test_project'
    var_7 = True
    var_8 = 'Existing content'
    var_9 = 'override'
    var_10 = 'project_name'
    var_11 = 'override_project'
    var_12 = {var_10: var_11}
    var_13 = 'binary.bin'
    var_14 = b'\x00\x01\x02\x03'
    var_15 = '{"project_name": "test_project", "_copy_without_render": ["*.bin"]}'
    var_16 = 'binary'



# Parsed testcases at query #53
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'Hello {{cookiecutter.project_name}}'
    assert var_5 == 'Hello test_project'
    var_6 = 'output'
    var_7 = 'test_project'
    var_8 = 'test.txt'
    var_9 = 'template'
    var_10 = 'project_name'
    var_11 = 'test_project'
    var_12 = {var_10: var_11}
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'output'
    var_15 = 'test_project'
    var_16 = 'existing'
    var_17 = True
    var_18 = 'template'
    var_19 = 'project_name'
    var_20 = 'test_project'
    var_21 = {var_19: var_20}
    var_22 = '{{cookiecutter.project_name}}'
    var_23 = 'Hello {{cookiecutter.project_name}}'
    var_24 = 'output'
    var_25 = 'test_project'
    var_26 = 'existing'
    assert var_26 == 'existing'
    var_27 = True
    var_28 = 'template'
    var_29 = 'project_name'
    var_30 = 'test_project'
    var_31 = {var_29: var_30}
    var_32 = 'hooks'
    var_33 = 'print("Pre hook executed")'
    var_34 = 'print("Post hook executed")'
    var_35 = '{{cookiecutter.project_name}}'
    var_36 = 'output'
    var_37 = True
    var_38 = 'test_project'
    var_39 = 'template'
    var_40 = 'project_name'
    var_41 = 'test_project'
    var_42 = {var_40: var_41}
    var_43 = '{{cookiecutter.project_name}}'
    var_44 = 'Hello {{cookiecutter.undefined_var}}'
    var_45 = 'output'
    var_46 = True
    var_47 = 'test_project'



# Parsed testcases at query #54
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



# Parsed testcases at query #55
#--------------------------


def test_case_0():
    var_0 = 'tests/mocks/valid-template'
    var_1 = 'tests/out'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = '_copy_without_render'
    var_6 = 'test_project'
    var_7 = '*.py'
    var_8 = [var_7]
    var_9 = {var_3: var_6, var_4: var_6, var_5: var_8}
    var_10 = {var_2: var_9}
    var_11 = True
    var_12 = 'README.md'



# Parsed testcases at query #56
#--------------------------


import jinja2.loaders as module_0
import jinja2.environment as module_1
import cookiecutter.generate as module_2
import cookiecutter.utils as module_3

def test_case_0():
    var_0 = '/tmp/test_project'
    var_1 = True
    var_2 = 'Hello, {{ name }}!'
    var_3 = 'test_template.txt'
    var_4 = '.'
    var_5 = module_0.FileSystemLoader(var_4)
    var_6 = module_1.Environment(loader=var_5)
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = '_copy_without_render'
    var_10 = '_new_lines'
    var_11 = []
    var_12 = '\n'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = 'World'
    var_15 = {var_7: var_13, var_8: var_14}
    var_16 = module_2.generate_file(var_0, var_3, var_15, var_6)
    var_17 = 'test_template.txt'
    var_18 = module_3.rmtree(var_0)



# Parsed testcases at query #57
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"project_name": "test"}'
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'cookiecutter'
    var_15 = 'project_name'
    var_16 = 'override'
    var_17 = {var_15: var_16}
    var_18 = {var_14: var_17}
    var_19 = True
    var_20 = 'template'
    var_21 = 'output'
    var_22 = 'cookiecutter.json'
    var_23 = '{"project_name": "test"}'
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'file.txt'
    var_26 = 'original'
    var_27 = True
    var_28 = 'modified'
    var_29 = 'test'
    var_30 = 'template'
    var_31 = 'output'
    var_32 = 'cookiecutter.json'
    var_33 = '{"project_name": "test"}'
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'hooks'
    var_36 = 'pre_gen_project.py'
    var_37 = var_25 / var_36
    var_38 = "print('pre hook')"
    var_39 = 'post_gen_project.py'
    var_40 = var_27 / var_39
    var_41 = "print('post hook')"
    var_42 = True
    var_43 = 'test'
    var_44 = 'template'
    var_45 = 'output'
    var_46 = 'cookiecutter.json'
    var_47 = '{"project_name": "test"}'
    var_48 = '{{cookiecutter.undefined_var}}'
    var_49 = True



# Parsed testcases at query #58
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = var_14 / var_17
    var_21 = 'existing_file'
    var_22 = var_20 / var_21
    var_23 = 'content'
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'cookiecutter'
    var_26 = {var_25: var_18}
    var_27 = True
    var_28 = var_20 / var_21
    var_29 = 'project_name'
    var_30 = ''
    var_31 = {var_29: var_30}
    var_32 = module_0.Environment()
    var_33 = '{{cookiecutter.project_name}}'
    var_34 = 'cookiecutter'
    var_35 = {var_34: var_31}
    var_36 = 'project_name'
    var_37 = 'version'
    var_38 = 'my_project'
    var_39 = '1.0'
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = module_0.Environment()
    var_42 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_43 = 'cookiecutter'
    var_44 = {var_43: var_40}



# Parsed testcases at query #59
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = '{{cookiecutter.project_name}}'
    var_9 = 'Hello, {{cookiecutter.author}}!'
    assert var_9 == 'Hello, Test Author!'
    var_10 = module_0.generate_files(var_0, var_5, var_6)
    var_11 = 'test.txt'
    var_12 = module_1.rmtree(var_0)
    var_13 = module_1.rmtree(var_6)



# Parsed testcases at query #60
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'Hello {{cookiecutter.project_name}}'
    assert var_5 == 'Hello test'
    var_6 = 'output'
    var_7 = 'test.txt'
    var_8 = 'template'
    var_9 = 'project_name'
    var_10 = 'test'
    var_11 = {var_9: var_10}
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'Hello {{cookiecutter.project_name}}'
    assert var_13 == 'Hello custom'
    var_14 = 'output'
    var_15 = 'project_name'
    var_16 = 'custom'
    var_17 = {var_15: var_16}
    var_18 = 'test.txt'
    var_19 = 'template'
    var_20 = 'project_name'
    var_21 = 'test'
    var_22 = {var_20: var_21}
    var_23 = '{{cookiecutter.project_name}}'
    var_24 = 'Hello {{cookiecutter.project_name}}'
    var_25 = 'output'
    var_26 = 'Modified'
    assert var_26 == 'Hello test'
    var_27 = True
    var_28 = 'template'
    var_29 = 'project_name'
    var_30 = 'test'
    var_31 = {var_29: var_30}
    var_32 = '{{cookiecutter.project_name}}'
    var_33 = 'Hello {{cookiecutter.project_name}}'
    var_34 = 'output'
    var_35 = 'Modified'
    assert var_35 == 'Modified'
    var_36 = True
    var_37 = 'template'
    var_38 = 'project_name'
    var_39 = 'test'
    var_40 = {var_38: var_39}
    var_41 = '{{cookiecutter.project_name}}'
    var_42 = 'Hello {{cookiecutter.project_name}}'
    var_43 = 'hooks'
    var_44 = 'print("Pre hook executed")'
    var_45 = 'print("Post hook executed")'
    var_46 = 'output'
    var_47 = True
    var_48 = 'test.txt'
    var_49 = 'template'
    var_50 = 'project_name'
    var_51 = 'test'
    var_52 = {var_50: var_51}
    var_53 = '{{cookiecutter.project_name}}'
    var_54 = 'Hello {{cookiecutter.undefined_var}}'
    var_55 = 'output'
    var_56 = 'template'
    var_57 = 'project_name'
    var_58 = 'test'
    var_59 = {var_57: var_58}
    var_60 = '{{cookiecutter.project_name}}'
    var_61 = 'binary.bin'
    var_62 = b'test binary content'
    assert var_62 == b'test binary content'
    var_63 = 'output'
    var_64 = 'template'
    var_65 = 'project_name'
    var_66 = '_copy_without_render'
    var_67 = 'test'
    var_68 = '*.md'
    var_69 = [var_68]
    var_70 = {var_65: var_67, var_66: var_69}
    var_71 = '{{cookiecutter.project_name}}'
    var_72 = 'Hello {{cookiecutter.project_name}}'
    assert var_72 == 'Hello {{cookiecutter.project_name}}'
    var_73 = 'output'
    var_74 = 'test.md'



# Parsed testcases at query #61
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'project_name'
    var_3 = 'test_project'
    var_4 = {var_2: var_3}
    var_5 = '{{cookiecutter.project_name}}'
    var_6 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_6 == 'Hello, test_project!'
    var_7 = 'test_project'
    var_8 = 'test.txt'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'project_name'
    var_12 = 'default'
    var_13 = {var_11: var_12}
    var_14 = '{{cookiecutter.project_name}}'
    var_15 = 'Hello, {{cookiecutter.project_name}}!'
    assert var_15 == 'Hello, custom!'
    var_16 = 'project_name'
    var_17 = 'custom'
    var_18 = {var_16: var_17}
    var_19 = 'test.txt'
    var_20 = 'template'
    var_21 = 'output'
    var_22 = 'project_name'
    var_23 = 'test'
    var_24 = {var_22: var_23}
    var_25 = '{{cookiecutter.project_name}}'
    var_26 = 'First version'
    var_27 = 'Second version'
    assert var_27 == 'Second version'
    var_28 = True
    var_29 = 'template'
    var_30 = 'output'
    var_31 = 'project_name'
    var_32 = 'test'
    var_33 = {var_31: var_32}
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'Original content'
    var_36 = 'New content'
    assert var_36 == 'Original content'
    var_37 = True
    var_38 = 'template'
    var_39 = 'output'
    var_40 = 'project_name'
    var_41 = 'test'
    var_42 = {var_40: var_41}
    var_43 = 'hooks'
    var_44 = 'print("Pre hook executed")'
    var_45 = '{{cookiecutter.project_name}}'
    var_46 = 'Content'
    var_47 = True
    var_48 = 'test'
    var_49 = 'test.txt'
    var_50 = 'template'
    var_51 = 'output'
    var_52 = 'project_name'
    var_53 = 'test'
    var_54 = {var_52: var_53}
    var_55 = '{{cookiecutter.project_name}}'
    var_56 = b'\x00\x01\x02\x03'
    var_57 = 'template'
    var_58 = 'output'
    var_59 = 'project_name'
    var_60 = '_copy_without_render'
    var_61 = 'test'
    var_62 = '*.bin'
    var_63 = [var_62]
    var_64 = {var_59: var_61, var_60: var_63}
    var_65 = '{{cookiecutter.project_name}}'
    var_66 = '{{cookiecutter.project_name}}'
    assert var_66 == '{{cookiecutter.project_name}}'
    var_67 = 'template'
    var_68 = 'output'
    var_69 = 'project_name'
    var_70 = 'test'
    var_71 = {var_69: var_70}
    var_72 = '{{cookiecutter.project_name}}'
    var_73 = '{{undefined_variable}}'
    var_74 = 'template'
    var_75 = 'output'
    var_76 = 'project_name'
    var_77 = 'test'
    var_78 = {var_76: var_77}
    var_79 = 'test'



# Parsed testcases at query #62
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



# Parsed testcases at query #63
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project'
    var_1 = 'test_file.txt'
    var_2 = 'Hello, {{ name }}!'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = '\n'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'World'
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = module_0.Environment()
    var_13 = 'test_binary.bin'
    var_14 = b'\x00\x01\x02\x03'
    var_15 = True



# Parsed testcases at query #64
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/simple'
    var_1 = 'tests/test-output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = True
    var_9 = 'README.md'



# Parsed testcases at query #65
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test_var'
    var_2 = 'test_value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/test_output'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = '{{ test_var }}'
    var_8 = True
    var_9 = ''
    var_10 = False



# Parsed testcases at query #66
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/mocks/cookiecutter.json'
    var_1 = module_0.generate_context(var_0)
    var_2 = 'project_name'
    var_3 = 'Default Project'
    var_4 = {var_2: var_3}
    var_5 = module_0.generate_context(var_0, var_4)
    var_6 = 'author'
    var_7 = 'Jane Smith'
    var_8 = {var_6: var_7}
    var_9 = module_0.generate_context(var_0, extra_context=var_8)
    var_10 = 'tests/mocks/invalid.json'
    var_11 = module_0.generate_context(var_10)
    var_12 = 'tests/mocks/nonexistent.json'
    var_13 = module_0.generate_context(var_12)
    var_14 = 'tests/mocks/empty.json'
    var_15 = module_0.generate_context(var_14)
    var_16 = []
    var_17 = 'tests/mocks/nested.json'
    var_18 = module_0.generate_context(var_17)



# Parsed testcases at query #67
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



# Parsed testcases at query #68
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



# Parsed testcases at query #69
#--------------------------


import jinja2.environment as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'test'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = '/tmp/test_output'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = '{{ test }}'
    var_8 = True
    var_9 = ''
    var_10 = '/tmp/test_output/existing_dir'
    var_11 = True
    var_12 = 'test_dir'
    var_13 = False
    var_14 = module_1.rmtree(var_4)



# Parsed testcases at query #70
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'test_template'
    var_1 = 'project_name'
    var_2 = 'author'
    var_3 = 'test_project'
    var_4 = 'Test Author'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'test_output'
    var_7 = True
    var_8 = 'templates'
    var_9 = 'Hello, {{ cookiecutter.project_name }}!'
    var_10 = 'project_name'
    var_11 = 'default_project'
    var_12 = {var_10: var_11}
    var_13 = module_0.generate_files(var_0, var_5, var_6)
    var_14 = 'test.txt'
    var_15 = module_1.rmtree(var_0)
    var_16 = module_1.rmtree(var_6)



# Parsed testcases at query #71
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'test_dir'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = '/tmp'
    var_5 = module_0.Environment()
    var_6 = False
    var_7 = True
    var_8 = ''
    var_9 = '{{ cookiecutter.project_name }}'



# Parsed testcases at query #72
#--------------------------


import cookiecutter.generate as module_0
import cookiecutter.utils as module_1

def test_case_0():
    var_0 = 'tests/mocks/valid-template'
    var_1 = 'tests/mocks/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'test_project'
    var_5 = {var_3: var_4}
    var_6 = {var_2: var_5}
    var_7 = True
    var_8 = module_0.generate_files(var_0, var_6, var_1, var_7)
    var_9 = 'README.md'
    var_10 = 'setup.py'
    var_11 = 'src'
    var_12 = '__init__.py'
    var_13 = module_1.rmtree(var_8)



# Parsed testcases at query #73
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



# Parsed testcases at query #74
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
    assert var_11 == 'Hello custom!'
    var_12 = 'default'
    var_13 = {var_11: var_12}
    var_14 = 'Hello {{ cookiecutter.project_name }}!'
    var_15 = '{{cookiecutter.project_name}}'
    var_16 = 'project_name'
    var_17 = 'custom'
    var_18 = {var_16: var_17}
    var_19 = 'test.txt'
    var_20 = 'template'
    var_21 = 'output'
    var_22 = 'project_name'
    var_23 = '_copy_without_render'
    var_24 = 'test'
    var_25 = '*.txt'
    var_26 = [var_25]
    var_27 = {var_22: var_24, var_23: var_26}
    var_28 = 'Hello {{ cookiecutter.project_name }}!'
    var_29 = '{{cookiecutter.project_name}}'
    var_30 = 'test'
    var_31 = 'test.txt'
    var_32 = 'template'
    var_33 = 'output'
    var_34 = 'project_name'
    assert var_34 == 'Hello test!'
    var_35 = 'test'
    var_36 = {var_34: var_35}
    var_37 = 'Hello {{ cookiecutter.project_name }}!'
    var_38 = '{{cookiecutter.project_name}}'
    var_39 = True
    var_40 = 'test'
    var_41 = 'test.txt'
    var_42 = 'template'
    var_43 = 'output'
    var_44 = 'project_name'
    var_45 = 'test'
    var_46 = {var_44: var_45}
    var_47 = 'Hello {{ cookiecutter.project_name }}!'
    var_48 = '{{cookiecutter.project_name}}'
    var_49 = 'Modified content'
    assert var_49 == 'Modified content'
    var_50 = True
    var_51 = 'template'
    var_52 = 'output'
    var_53 = 'project_name'
    var_54 = 'test'
    var_55 = {var_53: var_54}
    var_56 = 'hooks'
    var_57 = 'print("Pre hook executed")'
    var_58 = 'print("Post hook executed")'
    assert var_58 == 'Hello test!'
    var_59 = 'Hello {{ cookiecutter.project_name }}!'
    var_60 = '{{cookiecutter.project_name}}'
    var_61 = True
    var_62 = 'test'
    var_63 = 'test.txt'
    var_64 = 'template'
    var_65 = 'output'
    var_66 = 'project_name'
    var_67 = 'test'
    var_68 = {var_66: var_67}
    var_69 = b'\x00\x01\x02\x03'
    var_70 = '{{cookiecutter.project_name}}'
    var_71 = 'test'
    var_72 = 'binary.bin'
    var_73 = 'template'



# Parsed testcases at query #75
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test_project", "project_slug": "{{ cookiecutter.project_name.lower().replace(\' \', \'_\') }}"}'
    var_3 = '{{ cookiecutter.project_slug }}.txt'
    var_4 = 'Hello, {{ cookiecutter.project_name }}!'
    var_5 = 'copy_only.txt'
    var_6 = 'This file should be copied without rendering.'
    var_7 = 'copy_only_dir'
    var_8 = 'file.txt'
    var_9 = 'This file is in a copy-only directory.'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = '_copy_without_render'
    var_13 = 'Test Project'
    var_14 = 'copy_only_dir/*'
    var_15 = [var_5, var_14]
    var_16 = {var_11: var_13, var_12: var_15}
    var_17 = {var_10: var_16}
    var_18 = 'output'
    var_19 = True
    var_20 = False
    var_21 = 'test_project.txt'



# Parsed testcases at query #76
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = '{"project_name": "test"}'
    var_3 = '{{cookiecutter.project_name}}'
    var_4 = 'file.txt'
    var_5 = 'content'
    var_6 = 'output'
    var_7 = 'test'
    var_8 = 'template'
    var_9 = 'cookiecutter.json'
    var_10 = '{"project_name": "test"}'
    var_11 = '{{cookiecutter.project_name}}'
    var_12 = 'output'
    var_13 = 'test'
    var_14 = True
    var_15 = 'template'
    var_16 = 'cookiecutter.json'
    var_17 = '{"project_name": "test"}'
    var_18 = '{{cookiecutter.project_name}}'
    var_19 = 'file.txt'
    var_20 = 'content'
    var_21 = 'output'
    var_22 = 'test'
    var_23 = 'existing'
    var_24 = True
    var_25 = 'template'
    var_26 = 'cookiecutter.json'
    var_27 = '{"project_name": "test"}'
    var_28 = '{{cookiecutter.project_name}}'
    var_29 = 'hooks'
    var_30 = 'pre_gen_project.py'
    var_31 = var_19 / var_30
    var_32 = "print('pre')"
    var_33 = 'post_gen_project.py'
    var_34 = var_21 / var_33
    var_35 = "print('post')"
    var_36 = 'output'
    assert var_36 == '# {{cookiecutter.project_name}}'
    var_37 = True
    var_38 = 'test'
    var_39 = 'template'
    var_40 = 'cookiecutter.json'
    var_41 = '{"project_name": "default"}'
    var_42 = '{{cookiecutter.project_name}}'
    var_43 = 'output'
    var_44 = var_29 / var_43
    var_45 = 'project_name'
    var_46 = 'custom'
    var_47 = {var_45: var_46}
    var_48 = str(var_44)
    var_49 = module_0.generate_files(var_32, var_47, var_48)
    var_50 = var_44 / var_46
    var_51 = 'template'
    var_52 = 'cookiecutter.json'
    var_53 = '{"project_name": "test"}'
    var_54 = '{{cookiecutter.project_name}}'
    var_55 = b'\x00\x01\x02\x03'
    var_56 = 'binary.bin'
    var_57 = var_29 / var_56
    var_58 = 'output'
    var_59 = var_32 / var_58
    var_60 = str(var_59)
    var_61 = module_0.generate_files(var_50, output_dir=var_60)
    var_62 = 'test'
    var_63 = var_59 / var_62
    var_64 = var_63 / var_56
    var_65 = 'template'
    var_66 = 'cookiecutter.json'
    var_67 = '{"project_name": "test", "_copy_without_render": ["*.md"]}'
    var_68 = '{{cookiecutter.project_name}}'
    var_69 = 'readme.md'
    var_70 = var_29 / var_69
    var_71 = '# {{cookiecutter.project_name}}'
    var_72 = 'output'
    var_73 = var_58 / var_72
    var_74 = str(var_73)
    var_75 = module_0.generate_files(var_60, output_dir=var_74)
    var_76 = 'test'
    var_77 = var_73 / var_76
    var_78 = var_77 / var_69



# Parsed testcases at query #77
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = var_14 / var_17
    var_21 = '{{cookiecutter.project_name}}'
    var_22 = 'cookiecutter'
    var_23 = {var_22: var_18}
    var_24 = True
    var_25 = 'project_name'
    var_26 = ''
    var_27 = {var_25: var_26}
    var_28 = module_0.Environment()
    var_29 = '{{cookiecutter.project_name}}'
    var_30 = 'cookiecutter'
    var_31 = {var_30: var_27}
    var_32 = 'project_name'
    var_33 = 'version'
    var_34 = 'my_project'
    var_35 = '1.0'
    var_36 = {var_32: var_34, var_33: var_35}
    var_37 = module_0.Environment()
    var_38 = '{{cookiecutter.project_name}}-{{cookiecutter.version}}'
    var_39 = 'cookiecutter'
    var_40 = {var_39: var_36}



# Parsed testcases at query #78
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'output'
    var_2 = 'cookiecutter.json'
    var_3 = '{"project_name": "test"}'
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'file.txt'
    var_6 = 'content'
    var_7 = True
    var_8 = 'test'
    var_9 = 'template'
    var_10 = 'output'
    var_11 = 'cookiecutter.json'
    var_12 = '{"project_name": "test"}'
    var_13 = '{{cookiecutter.project_name}}'
    var_14 = 'file.txt'
    var_15 = 'cookiecutter'
    var_16 = 'project_name'
    var_17 = 'override'
    var_18 = {var_16: var_17}
    var_19 = {var_15: var_18}
    var_20 = True
    var_21 = 'template'
    var_22 = 'output'
    var_23 = 'cookiecutter.json'
    var_24 = '{"project_name": "test"}'
    var_25 = '{{cookiecutter.project_name}}'
    var_26 = 'file.txt'
    var_27 = 'content'
    var_28 = True
    var_29 = 'test'
    var_30 = 'template'
    var_31 = 'output'
    var_32 = 'cookiecutter.json'
    var_33 = '{"project_name": "test"}'
    var_34 = '{{cookiecutter.project_name}}'
    var_35 = 'hooks'
    var_36 = 'pre_gen_project.py'
    var_37 = var_26 / var_36
    var_38 = "print('pre hook')"
    var_39 = 'post_gen_project.py'
    var_40 = var_28 / var_39
    var_41 = "print('post hook')"
    var_42 = True
    var_43 = 'test'
    var_44 = 'template'
    var_45 = 'output'
    var_46 = 'cookiecutter.json'
    var_47 = '{"project_name": "test"}'
    var_48 = '{{cookiecutter.project_name}}'
    var_49 = 'file.txt'
    var_50 = var_35 / var_49
    var_51 = '{{cookiecutter.undefined_var}}'
    var_52 = True



# Parsed testcases at query #79
#--------------------------


def test_case_0():
    var_0 = 'project'
    var_1 = 'Hello, {{ name }}!'
    var_2 = 'template.txt'
    var_3 = 'cookiecutter'
    var_4 = 'name'
    var_5 = '_copy_without_render'
    var_6 = '_new_lines'
    var_7 = []
    var_8 = None
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'World'
    var_11 = {var_3: var_9, var_4: var_10}
    var_12 = False
    var_13 = 'binary.bin'
    var_14 = b'\x00\x01\x02\x03'
    var_15 = True
    var_16 = 'empty_dir'
    var_17 = ''
    var_18 = False



# Parsed testcases at query #80
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



# Parsed testcases at query #81
#--------------------------


def test_case_0():
    var_0 = 'tests/test-templates/test-template'
    var_1 = 'tests/test-output'
    var_2 = 'project_name'
    var_3 = 'project_slug'
    var_4 = 'author'
    var_5 = 'test_project'
    var_6 = 'Test Author'
    var_7 = {var_2: var_5, var_3: var_5, var_4: var_6}
    var_8 = True
    var_9 = False
    var_10 = 'README.md'
    var_11 = 'setup.py'



# Parsed testcases at query #82
#--------------------------


def test_case_0():
    var_0 = 'tests/fake-repo-pre'
    var_1 = 'tests/output'
    var_2 = 'cookiecutter'
    var_3 = 'project_name'
    var_4 = 'project_slug'
    var_5 = 'test_project'
    var_6 = {var_3: var_5, var_4: var_5}
    var_7 = {var_2: var_6}
    var_8 = True
    var_9 = False
    var_10 = 'invalid_variable'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = {var_2: var_12}
    var_14 = 'non-existent-repo'



# Parsed testcases at query #83
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'tests/test_context.json'
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = 'project_slug'
    var_5 = 'test_slug'
    var_6 = {var_4: var_5}
    var_7 = module_0.generate_context(var_0, var_3, var_6)
    var_8 = 'tests/invalid_context.json'
    var_9 = module_0.generate_context(var_8)
    var_10 = 'tests/missing_context.json'
    var_11 = module_0.generate_context(var_10)
    var_12 = 'tests/empty_context.json'
    var_13 = module_0.generate_context(var_12)
    var_14 = module_0.generate_context(var_0)



# Parsed testcases at query #84
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'test_project'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{cookiecutter.project_name}}'
    var_5 = 'cookiecutter'
    var_6 = {var_5: var_2}
    var_7 = 'project_name'
    var_8 = 'test_project'
    var_9 = {var_7: var_8}
    var_10 = module_0.Environment()
    var_11 = var_4 / var_8
    var_12 = '{{cookiecutter.project_name}}'
    var_13 = 'cookiecutter'
    var_14 = {var_13: var_9}
    var_15 = False
    var_16 = 'project_name'
    var_17 = 'test_project'
    var_18 = {var_16: var_17}
    var_19 = module_0.Environment()
    var_20 = var_14 / var_17
    var_21 = 'test.txt'
    var_22 = var_20 / var_21
    var_23 = 'old content'
    var_24 = '{{cookiecutter.project_name}}'
    var_25 = 'cookiecutter'
    var_26 = {var_25: var_18}
    var_27 = True
    var_28 = 'project_name'
    var_29 = ''
    var_30 = {var_28: var_29}
    var_31 = module_0.Environment()
    var_32 = '{{cookiecutter.project_name}}'
    var_33 = 'cookiecutter'
    var_34 = {var_33: var_30}
    var_35 = 'project_name'
    var_36 = 'nested'
    var_37 = 'test_project'
    var_38 = 'nested_dir'
    var_39 = {var_35: var_37, var_36: var_38}
    var_40 = module_0.Environment()
    var_41 = '{{cookiecutter.project_name}}/{{cookiecutter.nested}}'
    var_42 = 'cookiecutter'
    var_43 = {var_42: var_39}



