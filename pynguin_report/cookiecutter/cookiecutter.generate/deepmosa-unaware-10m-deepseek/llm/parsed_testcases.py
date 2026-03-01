####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'TestProject'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{cookiecutter.project_name}}.txt'
    var_7 = 'Version: {{cookiecutter.version}}'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = '_new_lines'
    var_11 = '\n'
    var_12 = {var_1: var_3, var_2: var_4, var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = False
    var_15 = 'TestProject.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'test.txt'
    var_5 = 'Content'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = '_new_lines'
    var_10 = 'Test'
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False
    var_15 = 'Updated Content'
    var_16 = False
    var_17 = module_0.generate_files(var_1, var_13, var_2, var_16, accept_hooks=var_16)
    var_18 = True

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'test.txt'
    var_5 = 'Original'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'name'
    var_9 = '_new_lines'
    var_10 = 'Test'
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False
    var_15 = 'Modified'
    var_16 = True

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = '_copy_without_render'
    var_3 = 'Test'
    var_4 = '*.bin'
    var_5 = 'data/*'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'test.bin'
    var_9 = b'binary content'
    var_10 = 'data'
    var_11 = 'info.txt'
    var_12 = '{{should_not_render}}'
    var_13 = '{{cookiecutter.name}}.txt'
    var_14 = 'Name: {{cookiecutter.name}}'
    var_15 = 'output'
    var_16 = 'cookiecutter'
    var_17 = '_new_lines'
    var_18 = 'TestProject'
    var_19 = [var_4, var_5]
    var_20 = '\n'
    var_21 = {var_1: var_18, var_2: var_19, var_17: var_20}
    var_22 = {var_16: var_21}
    var_23 = False
    var_24 = 'TestProject.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = "\nimport os\nwith open('pre_hook_ran.txt', 'w') as f:\n    f.write('pre hook executed')\n"
    var_7 = 'post_gen_project.py'
    var_8 = "\nimport os\nwith open('post_hook_ran.txt', 'w') as f:\n    f.write('post hook executed')\n"
    var_9 = 'test.txt'
    var_10 = 'Content'
    var_11 = 'output'
    var_12 = 'cookiecutter'
    var_13 = 'name'
    var_14 = '_new_lines'
    var_15 = 'Test'
    var_16 = '\n'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = {var_12: var_17}
    var_19 = True
    var_20 = module_0.generate_files(var_1, var_18, var_2, accept_hooks=var_19)

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}



# Parsed testcases at query #2
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'existing'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'new_var'
    var_12 = 'new_value'
    var_13 = {var_11: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_10, var_13)
    var_15 = 'choices'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = 'd'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = {var_15: var_20}
    var_22 = [var_17, var_18]
    var_23 = {var_15: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = [var_16, var_17, var_18]
    var_26 = {var_15: var_25}
    var_27 = [var_17, var_19]
    var_28 = {var_15: var_27}
    var_29 = module_0.apply_overwrites_to_context(var_26, var_28)
    var_30 = 'choice'
    var_31 = 'first'
    var_32 = 'second'
    var_33 = 'third'
    var_34 = [var_31, var_32, var_33]
    var_35 = {var_30: var_34}
    var_36 = {var_30: var_32}
    var_37 = module_0.apply_overwrites_to_context(var_35, var_36)
    var_38 = [var_16, var_17, var_18]
    var_39 = {var_30: var_38}
    var_40 = {var_30: var_19}
    var_41 = module_0.apply_overwrites_to_context(var_39, var_40)
    var_42 = 'config'
    var_43 = 'key1'
    var_44 = 'key2'
    var_45 = 'key3'
    var_46 = 'val1'
    var_47 = 'val2'
    var_48 = 'val3'
    var_49 = {var_43: var_46, var_44: var_47, var_45: var_48}
    var_50 = {var_42: var_49}
    var_51 = 'key4'
    var_52 = 'new_val2'
    var_53 = 'val4'
    var_54 = {var_44: var_52, var_51: var_53}
    var_55 = {var_42: var_54}
    var_56 = module_0.apply_overwrites_to_context(var_50, var_55)
    var_57 = 'flag'
    var_58 = False
    var_59 = {var_57: var_58}
    var_60 = 'yes'
    var_61 = {var_57: var_60}
    var_62 = module_0.apply_overwrites_to_context(var_59, var_61)
    var_63 = True
    var_64 = {var_57: var_63}
    var_65 = 'no'
    var_66 = {var_57: var_65}
    var_67 = module_0.apply_overwrites_to_context(var_64, var_66)
    var_68 = {var_57: var_63}
    var_69 = 'maybe'
    var_70 = {var_57: var_69}
    var_71 = module_0.apply_overwrites_to_context(var_68, var_70)
    var_72 = 'deep'
    var_73 = 'level1'
    var_74 = 'level2'
    var_75 = {var_74: var_2}
    var_76 = {var_73: var_75}
    var_77 = {var_72: var_76}
    var_78 = 'level2_new'
    var_79 = 'added'
    var_80 = {var_74: var_5, var_78: var_79}
    var_81 = {var_73: var_80}
    var_82 = {var_72: var_81}
    var_83 = module_0.apply_overwrites_to_context(var_77, var_82, in_dictionary_variable=var_63)
    var_84 = 'nested'
    var_85 = {var_84: var_9}
    var_86 = {var_8: var_85}
    var_87 = 'new_key'
    var_88 = {var_87: var_12}
    var_89 = module_0.apply_overwrites_to_context(var_86, var_88, in_dictionary_variable=var_63)



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old_value'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_value'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'existing'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'new_var'
    var_12 = {var_11: var_5}
    var_13 = module_0.apply_overwrites_to_context(var_10, var_12)
    var_14 = 'choice_var'
    var_15 = 'option1'
    var_16 = 'option2'
    var_17 = 'option3'
    var_18 = [var_15, var_16, var_17]
    var_19 = {var_14: var_18}
    var_20 = {var_14: var_16}
    var_21 = module_0.apply_overwrites_to_context(var_19, var_20)
    var_22 = [var_15, var_16]
    var_23 = {var_14: var_22}
    var_24 = 'invalid_option'
    var_25 = {var_14: var_24}
    var_26 = module_0.apply_overwrites_to_context(var_23, var_25)
    var_27 = 'multichoice'
    var_28 = 'a'
    var_29 = 'b'
    var_30 = 'c'
    var_31 = 'd'
    var_32 = [var_28, var_29, var_30, var_31]
    var_33 = {var_27: var_32}
    var_34 = [var_29, var_30]
    var_35 = {var_27: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_33, var_35)
    var_37 = [var_28, var_29, var_30]
    var_38 = {var_27: var_37}
    var_39 = [var_28, var_31]
    var_40 = {var_27: var_39}
    var_41 = module_0.apply_overwrites_to_context(var_38, var_40)
    var_42 = 'nested'
    var_43 = 'key1'
    var_44 = 'key2'
    var_45 = 'key3'
    var_46 = 'value1'
    var_47 = 'value2'
    var_48 = 'value3'
    var_49 = {var_43: var_46, var_44: var_47, var_45: var_48}
    var_50 = {var_42: var_49}
    var_51 = 'key4'
    var_52 = 'updated'
    var_53 = {var_44: var_52, var_51: var_5}
    var_54 = {var_42: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_50, var_54)
    var_56 = 'flag'
    var_57 = False
    var_58 = {var_56: var_57}
    var_59 = 'yes'
    var_60 = {var_56: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_58, var_60)
    var_62 = True
    var_63 = {var_56: var_62}
    var_64 = 'no'
    var_65 = {var_56: var_64}
    var_66 = module_0.apply_overwrites_to_context(var_63, var_65)
    var_67 = {var_56: var_62}
    var_68 = 'invalid'
    var_69 = {var_56: var_68}
    var_70 = module_0.apply_overwrites_to_context(var_67, var_69)
    var_71 = {var_8: var_9}
    var_72 = {var_42: var_71}
    var_73 = 'new_key'
    var_74 = {var_73: var_5}
    var_75 = {var_42: var_74}
    var_76 = module_0.apply_overwrites_to_context(var_72, var_75, in_dictionary_variable=var_62)
    var_77 = 'top'
    var_78 = 'config'
    var_79 = 'enabled'
    var_80 = 'choices'
    var_81 = 'settings'
    var_82 = 'opt1'
    var_83 = 'opt2'
    var_84 = 'opt3'
    var_85 = [var_82, var_83, var_84]
    var_86 = 'timeout'
    var_87 = 'retries'
    var_88 = 30
    var_89 = 3
    var_90 = {var_86: var_88, var_87: var_89}
    var_91 = {var_79: var_62, var_80: var_85, var_81: var_90}
    var_92 = {var_77: var_9, var_78: var_91}
    var_93 = 'new_setting'
    var_94 = 60
    var_95 = 'added'
    var_96 = {var_86: var_94, var_93: var_95}
    var_97 = {var_79: var_64, var_80: var_83, var_81: var_96}
    var_98 = {var_77: var_52, var_78: var_97}
    var_99 = module_0.apply_overwrites_to_context(var_92, var_98)



# Parsed testcases at query #4
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'MyProject'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}_app'
    var_5 = 'MyProject_app'
    var_6 = 'project_name'
    var_7 = 'ExistingProject'
    var_8 = {var_6: var_7}
    var_9 = module_0.Environment()
    var_10 = '{{ project_name }}'
    var_11 = 'project_name'
    var_12 = 'ExistingProject'
    var_13 = {var_11: var_12}
    var_14 = module_0.Environment()
    var_15 = '{{ project_name }}'
    var_16 = True
    var_17 = {}
    var_18 = module_0.Environment()
    var_19 = ''
    var_20 = 'author'
    var_21 = 'year'
    var_22 = 'version'
    var_23 = 'John Doe'
    var_24 = '2023'
    var_25 = '1.0'
    var_26 = {var_20: var_23, var_21: var_24, var_22: var_25}
    var_27 = module_0.Environment()
    var_28 = 'project_{{ author }}_{{ year }}_{{ version }}'
    var_29 = 'project_John Doe_2023_1.0'
    var_30 = 'module'
    var_31 = 'utils'
    var_32 = {var_30: var_31}
    var_33 = module_0.Environment()
    var_34 = 'src/{{ module }}/tests'
    var_35 = 'src/utils/tests'
    var_36 = 'src'
    var_37 = 'src/utils'
    var_38 = 'name'
    var_39 = 'my-project'
    var_40 = {var_38: var_39}
    var_41 = module_0.Environment()
    var_42 = "{{ name|upper|replace('-', '_') }}"
    var_43 = 'MY_PROJECT'



# Parsed testcases at query #5
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'TestProject'
    var_4 = '1.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'cookiecutter.json'
    var_7 = module_0.dumps(var_5)
    var_8 = '{{cookiecutter.project_name}}.txt'
    var_9 = 'Version: {{cookiecutter.version}}'
    var_10 = 'output'
    var_11 = 'cookiecutter'
    var_12 = {var_11: var_5}
    var_13 = False
    var_14 = 'TestProject.txt'

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'test.txt'
    var_7 = 'content'
    var_8 = 'output'
    var_9 = 'existing.txt'
    var_10 = 'old content'
    var_11 = False
    var_12 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'test.txt'
    var_7 = 'new content'
    var_8 = 'output'
    var_9 = 'old content'
    var_10 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'name'
    var_2 = '_copy_without_render'
    var_3 = 'Test'
    var_4 = '*.bin'
    var_5 = 'data/*'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'cookiecutter.json'
    var_9 = module_0.dumps(var_7)
    var_10 = 'test.bin'
    var_11 = b'binary\x00content'
    var_12 = 'data'
    var_13 = 'info.dat'
    var_14 = b'data\x00file'
    var_15 = '{{cookiecutter.name}}.txt'
    var_16 = 'Name: {{cookiecutter.name}}'
    var_17 = 'output'
    var_18 = 'cookiecutter'
    var_19 = {var_18: var_7}
    var_20 = False
    var_21 = 'Test.txt'

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '{{undefined_var}}.txt'
    var_7 = 'content'
    var_8 = 'output'
    var_9 = False

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = 'test.txt'
    var_7 = 'content'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = "#!/usr/bin/env python\nprint('pre hook')"
    var_11 = 493
    var_12 = 'output'
    var_13 = True
    var_14 = 'pre_gen_project'
    var_15 = 'Test'
    var_16 = var_4 / var_15
    var_17 = str(var_16)
    var_18 = 'cookiecutter'
    var_19 = 'name'
    var_20 = {var_19: var_15}
    var_21 = {var_18: var_20}

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '{{bad_template}.txt'
    var_7 = 'content'
    var_8 = 'output'
    var_9 = False
    var_10 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'name'
    var_3 = 'Test'
    var_4 = {var_2: var_3}
    var_5 = module_0.dumps(var_4)
    var_6 = '{{bad_template}.txt'
    var_7 = 'content'
    var_8 = 'output'
    var_9 = False
    var_10 = True



# Parsed testcases at query #6
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = '{{ project_name }}.txt'
    var_2 = 'Hello {{ name }}!'
    var_3 = 'cookiecutter.json'
    var_4 = '{"project_name": "test_project", "name": "World"}'
    var_5 = 'output'
    var_6 = False
    var_7 = 'test_project'
    var_8 = 'test_project.txt'
    var_9 = 'template'
    var_10 = 'render.txt'
    var_11 = 'Render {{ var }}'
    var_12 = 'copy.txt'
    var_13 = 'Copy {{ var }}'
    var_14 = 'cookiecutter.json'
    var_15 = '{"project_name": "test", "var": "value", "_copy_without_render": ["copy.txt"]}'
    var_16 = 'output'
    var_17 = 'template'
    var_18 = 'test.txt'
    var_19 = 'Content'
    var_20 = 'cookiecutter.json'
    var_21 = '{"project_name": "project"}'
    var_22 = 'output'
    var_23 = var_13 / var_22
    var_24 = module_0.generate_context(var_15)
    var_25 = str(var_23)
    var_26 = False
    var_27 = str(var_23)
    var_28 = False
    var_29 = str(var_23)
    var_30 = True
    var_31 = module_0.generate_files(var_7, var_24, var_29, var_30)
    var_32 = 'template'
    var_33 = 'test.txt'
    var_34 = 'Original'
    var_35 = 'cookiecutter.json'
    var_36 = '{"project_name": "project"}'
    var_37 = 'output'
    var_38 = var_13 / var_37
    var_39 = module_0.generate_context(var_15)
    var_40 = 'project'
    var_41 = var_38 / var_40
    var_42 = var_41 / var_33
    var_43 = 'Manual content'
    var_44 = str(var_38)
    var_45 = True
    var_46 = module_0.generate_files(var_29, var_39, var_44, skip_if_file_exists=var_45)
    var_47 = 'template'
    var_48 = 'binary.dat'
    var_49 = b'\x00\x01\x02\x03\x04'
    var_50 = 'text.txt'
    var_51 = 'Text {{ var }}'
    var_52 = 'cookiecutter.json'
    var_53 = '{"project_name": "project", "var": "rendered"}'
    var_54 = 'output'
    var_55 = var_15 / var_54
    var_56 = module_0.generate_context(var_43)
    var_57 = str(var_55)
    var_58 = module_0.generate_files(var_7, var_56, var_57)
    var_59 = var_44 / var_48
    var_60 = var_45 / var_50
    var_61 = 'template'
    var_62 = 'subdir'
    var_63 = 'nested'
    var_64 = 'root.txt'
    var_65 = 'Root'
    var_66 = 'sub.txt'
    var_67 = 'Sub {{ var }}'
    var_68 = 'nested.txt'
    var_69 = 'Nested'
    var_70 = 'cookiecutter.json'
    var_71 = '{"project_name": "project", "var": "value"}'
    var_72 = 'output'
    var_73 = str(var_55)
    var_74 = 'template'
    var_75 = 'hooks'
    var_76 = 'pre_gen_project.py'
    var_77 = "#!/usr/bin/env python\nprint('pre hook')"
    var_78 = 493
    var_79 = 'test.txt'
    var_80 = 'Content'
    var_81 = 'cookiecutter.json'
    var_82 = '{"project_name": "project"}'
    var_83 = 'output'
    var_84 = var_7 / var_83
    var_85 = module_0.generate_context(var_70)
    var_86 = str(var_84)
    var_87 = False
    var_88 = module_0.generate_files(var_71, var_85, var_86, accept_hooks=var_87)
    var_89 = 'template'
    var_90 = 'simple.txt'
    var_91 = 'Simple content'
    var_92 = 'cookiecutter.json'
    var_93 = '{"project_name": "project"}'
    var_94 = 'output'
    var_95 = var_65 / var_94
    var_96 = None
    var_97 = str(var_95)
    var_98 = module_0.generate_files(var_15, var_96, var_97)



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'A'
    var_8 = 'B'
    var_9 = 'C'
    var_10 = [var_7, var_8, var_9]
    var_11 = '1.0.0'
    var_12 = {var_5: var_10, var_6: var_11}
    var_13 = 'project_name'
    var_14 = 'B'
    var_15 = {var_13: var_14}
    var_16 = 'project_name'
    var_17 = 'version'
    var_18 = 'Test'
    var_19 = '1.0.0'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = 'project_name'
    var_22 = 'Overridden'
    var_23 = {var_21: var_22}
    var_24 = '{"invalid": json}'
    var_25 = 'project'
    var_26 = 'list'
    var_27 = 'name'
    var_28 = 'settings'
    var_29 = 'Test'
    var_30 = 'debug'
    var_31 = True
    var_32 = {var_30: var_31}
    var_33 = {var_27: var_29, var_28: var_32}
    var_34 = 'a'
    var_35 = 'b'
    var_36 = 'c'
    var_37 = [var_34, var_35, var_36]
    var_38 = {var_25: var_33, var_26: var_37}
    var_39 = {}
    var_40 = 'choice'
    var_41 = 'A'
    var_42 = 'B'
    assert var_42 == 1
    var_43 = [var_41, var_42]
    var_44 = {var_40: var_43}
    var_45 = 'choice'
    var_46 = 'C'
    var_47 = {var_45: var_46}
    var_48 = 'always'
    var_49 = 0
    var_50 = var_44.message
    var_51 = str(var_50)



# Parsed testcases at query #8
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'template'
    var_1 = '{{project_name}}.txt'
    var_2 = 'Hello {{author}}!'
    var_3 = 'cookiecutter.json'
    var_4 = 'project_name'
    var_5 = 'author'
    var_6 = 'test_project'
    var_7 = 'Test Author'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = module_0.dumps(var_8)
    var_10 = 'test_project.txt'
    var_11 = 'template'
    var_12 = 'regular.txt'
    var_13 = 'Render {{variable}}'
    var_14 = 'copy.txt'
    var_15 = "Don't render {{variable}}"
    var_16 = 'cookiecutter.json'
    var_17 = 'project_name'
    var_18 = '_copy_without_render'
    var_19 = 'test'
    var_20 = [var_14]
    var_21 = {var_17: var_19, var_18: var_20}
    var_22 = module_0.dumps(var_21)
    var_23 = 'template'
    var_24 = 'test.txt'
    var_25 = 'content'
    var_26 = 'cookiecutter.json'
    var_27 = 'project_name'
    var_28 = 'myproject'
    var_29 = {var_27: var_28}
    var_30 = module_0.dumps(var_29)
    var_31 = module_1.generate_context(var_19)
    var_32 = var_20 / var_28
    var_33 = 'existing.txt'
    var_34 = var_32 / var_33
    var_35 = 'old content'
    var_36 = False
    var_37 = True
    assert var_37 == 'existing content'
    var_38 = 'template'
    var_39 = 'test.txt'
    var_40 = 'new content'
    var_41 = 'cookiecutter.json'
    var_42 = 'project_name'
    var_43 = 'project'
    var_44 = {var_42: var_43}
    var_45 = module_0.dumps(var_44)
    var_46 = module_1.generate_context(var_19)
    var_47 = var_20 / var_43
    var_48 = var_47 / var_39
    var_49 = 'existing content'
    var_50 = True
    var_51 = 'template'
    var_52 = 'image.png'
    var_53 = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
    var_54 = 'cookiecutter.json'
    var_55 = 'project_name'
    var_56 = 'test'
    var_57 = {var_55: var_56}
    var_58 = module_0.dumps(var_57)
    var_59 = module_1.generate_context(var_19)
    var_60 = var_21 / var_56
    var_61 = var_60 / var_52
    var_62 = 'template'
    var_63 = 'test.txt'
    var_64 = 'test'
    var_65 = 'cookiecutter.json'
    var_66 = 'project_name'
    var_67 = 'hook_test'
    var_68 = {var_66: var_67}
    var_69 = module_0.dumps(var_68)
    var_70 = 'hooks'
    var_71 = 'pre_gen_project.py'
    var_72 = "#!/usr/bin/env python\nprint('pre hook executed')"
    var_73 = 493
    var_74 = module_1.generate_context(var_37)
    var_75 = True
    var_76 = 'pre_gen_project'
    var_77 = 'hook_test'
    var_78 = var_64 / var_77
    var_79 = 'template'
    var_80 = '{{undefined_var}}.txt'
    var_81 = 'test'
    var_82 = 'cookiecutter.json'
    var_83 = 'project_name'
    var_84 = {var_83: var_81}
    var_85 = module_0.dumps(var_84)
    var_86 = module_1.generate_context(var_18)
    var_87 = 'template'
    var_88 = '{{undefined}}.txt'
    var_89 = 'test'
    var_90 = 'cookiecutter.json'
    var_91 = 'project_name'
    var_92 = {var_91: var_89}
    var_93 = module_0.dumps(var_92)
    var_94 = module_1.generate_context(var_18)
    var_95 = True



# Parsed testcases at query #9
#--------------------------


import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'TestProject'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'cookiecutter.json'
    var_7 = module_0.dumps(var_5)
    var_8 = '{{cookiecutter.project_name}}.txt'
    var_9 = 'Version: {{cookiecutter.version}}'
    var_10 = 'output'
    var_11 = 'cookiecutter'
    var_12 = {var_11: var_5}
    var_13 = False
    var_14 = 'TestProject.txt'

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = 'test.txt'
    var_7 = 'Content'
    var_8 = 'output'
    var_9 = 'old.txt'
    var_10 = 'Old content'
    var_11 = 'cookiecutter'
    var_12 = {var_11: var_3}
    var_13 = False
    var_14 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = 'test.txt'
    var_7 = 'New content'
    var_8 = 'output'
    var_9 = 'Existing content'
    var_10 = 'cookiecutter'
    var_11 = {var_10: var_3}
    var_12 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'TestProject'
    var_4 = 'static/*'
    var_5 = 'config.json'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'cookiecutter.json'
    var_9 = module_0.dumps(var_7)
    var_10 = 'static'
    var_11 = 'image.png'
    var_12 = b'binary data'
    var_13 = '{"key": "{{cookiecutter.project_name}}"}'
    var_14 = 'README.md'
    var_15 = '# {{cookiecutter.project_name}}'
    var_16 = 'output'
    var_17 = 'cookiecutter'
    var_18 = {var_17: var_7}

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = 'binary.dat'
    var_7 = b'\x00\x01\x02\x03\x04'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = {var_9: var_3}

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = '{{undefined_var}}.txt'
    var_7 = 'Content'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = {var_9: var_3}

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = 'test.txt'
    var_7 = 'Content'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = "\nimport sys\nsys.path.insert(0, '.')\nwith open('pre_hook_ran.txt', 'w') as f:\n    f.write('pre hook executed')\n"
    var_11 = 'output'
    var_12 = 'cookiecutter'
    var_13 = {var_12: var_3}
    var_14 = True

import json as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'TestProject'
    var_3 = {var_1: var_2}
    var_4 = 'cookiecutter.json'
    var_5 = module_0.dumps(var_3)
    var_6 = 'test.txt'
    var_7 = 'Content'
    var_8 = 'hooks'
    var_9 = 'pre_gen_project.py'
    var_10 = "\nimport sys\nsys.path.insert(0, '.')\nwith open('pre_hook_ran.txt', 'w') as f:\n    f.write('pre hook executed')\n"
    var_11 = 'output'
    var_12 = 'cookiecutter'
    var_13 = {var_12: var_3}
    var_14 = True



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'version'
    var_3 = 'TestProject'
    var_4 = '1.0.0'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '{{cookiecutter.project_name}}.txt'
    var_7 = 'Version: {{cookiecutter.version}}'
    var_8 = 'output'
    var_9 = 'cookiecutter'
    var_10 = '_new_lines'
    var_11 = '\n'
    var_12 = {var_1: var_3, var_2: var_4, var_10: var_11}
    var_13 = {var_9: var_12}
    var_14 = False
    var_15 = 'TestProject.txt'

import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'test.txt'
    var_5 = 'content'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_new_lines'
    var_10 = 'Test'
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False
    var_15 = 'new content'
    var_16 = False
    var_17 = module_0.generate_files(var_1, var_13, var_2, var_16)
    var_18 = True

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'test.txt'
    var_5 = 'original'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_new_lines'
    var_10 = 'Test'
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False
    var_15 = 'modified'
    var_16 = True

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = '_copy_without_render'
    var_3 = 'Test'
    var_4 = 'static/*'
    var_5 = 'config.json'
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}
    var_8 = 'static'
    var_9 = 'image.png'
    var_10 = b'binary data'
    var_11 = '{"key": "{{cookiecutter.project_name}}"}'
    var_12 = 'README.md'
    var_13 = '# {{cookiecutter.project_name}}'
    var_14 = 'output'
    var_15 = 'cookiecutter'
    var_16 = '_new_lines'
    var_17 = [var_4, var_5]
    var_18 = '\n'
    var_19 = {var_1: var_3, var_2: var_17, var_16: var_18}
    var_20 = {var_15: var_19}
    var_21 = False

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'binary.bin'
    var_5 = b'\x00\x01\x02\x03\x04'
    var_6 = 'output'
    var_7 = 'cookiecutter'
    var_8 = 'project_name'
    var_9 = '_new_lines'
    var_10 = 'Test'
    var_11 = '\n'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = {var_7: var_12}
    var_14 = False

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = 'hooks'
    var_5 = 'pre_gen_project.py'
    var_6 = "\nimport os\nwith open(os.path.join('{{cookiecutter.project_name}}', 'pre_hook.txt'), 'w') as f:\n    f.write('pre-hook executed')\n"
    var_7 = 'test.txt'
    var_8 = 'content'
    var_9 = 'output'
    var_10 = 'cookiecutter'
    var_11 = 'project_name'
    var_12 = '_new_lines'
    var_13 = 'Test'
    var_14 = '\n'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = {var_10: var_15}
    var_17 = False
    var_18 = True
    var_19 = 'pre_hook.txt'

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'template'
    var_1 = 'project_name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}



# Parsed testcases at query #11
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'version'
    var_11 = '2.0'
    var_12 = {var_10: var_11}
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'Test'
    var_16 = '1.0'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'version'
    var_19 = '3.0'
    var_20 = {var_18: var_19}
    var_21 = 'project_name'
    var_22 = 'version'
    var_23 = 'Test'
    var_24 = '1.0'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'version'
    var_27 = '2.0'
    var_28 = {var_26: var_27}
    var_29 = '3.0'
    var_30 = {var_26: var_29}
    var_31 = '{"invalid": json}'
    var_32 = 'project'
    var_33 = 'list'
    var_34 = 'name'
    var_35 = 'settings'
    var_36 = 'Test'
    var_37 = 'debug'
    var_38 = True
    var_39 = {var_37: var_38}
    var_40 = {var_34: var_36, var_35: var_39}
    var_41 = 'a'
    var_42 = 'b'
    var_43 = 'c'
    var_44 = [var_41, var_42, var_43]
    var_45 = {var_32: var_40, var_33: var_44}
    var_46 = 'test'
    var_47 = 'value'
    var_48 = {var_46: var_47}
    var_49 = 0
    var_50 = os.path.splitext(var_47)[var_49]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test Project'
    var_8 = '1.0.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'project_name'
    var_11 = 'Default Project'
    var_12 = {var_10: var_11}
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'Test Project'
    var_16 = '1.0.0'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'version'
    var_19 = '2.0.0'
    var_20 = {var_18: var_19}
    var_21 = 'project_name'
    var_22 = 'version'
    var_23 = 'Test Project'
    var_24 = '1.0.0'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'project_name'
    var_27 = 'Default Project'
    var_28 = {var_26: var_27}
    var_29 = 'version'
    var_30 = '3.0.0'
    var_31 = {var_29: var_30}
    var_32 = '{"invalid": json}'
    var_33 = False
    var_34 = True
    var_35 = False
    var_36 = 'non_existent_file.json'
    var_37 = module_0.generate_context(var_36)
    var_38 = True
    var_39 = 'project'
    var_40 = 'settings'
    var_41 = 'name'
    var_42 = 'author'
    var_43 = 'Test'
    var_44 = 'Developer'
    var_45 = {var_41: var_43, var_42: var_44}
    var_46 = 'option1'
    var_47 = 'option2'
    var_48 = [var_46, var_47]
    var_49 = {var_39: var_45, var_40: var_48}
    var_50 = {}



# Parsed testcases at query #2
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'myproject'
    var_3 = {var_1: var_2}
    var_4 = '{{ project_name }}'
    var_5 = ''
    var_6 = 'myproject'
    var_7 = True
    var_8 = 'user'
    var_9 = 'project'
    var_10 = 'john'
    var_11 = 'test'
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 'projects/{{ user }}/{{ project }}'
    var_14 = 'projects/john/test'
    var_15 = 'level1'
    var_16 = 'level2'
    var_17 = 'level3'
    var_18 = 'a'
    var_19 = 'b'
    var_20 = 'c'
    var_21 = {var_15: var_18, var_16: var_19, var_17: var_20}
    var_22 = '{{ level1 }}/{{ level2 }}/{{ level3 }}'
    var_23 = 'a/b/c'
    var_24 = 'name'
    var_25 = 'test-project_123'
    var_26 = {var_24: var_25}
    var_27 = '{{ name }}'



# Parsed testcases at query #3
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'original'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new_name'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = 'existing'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = 'new_var'
    var_12 = 'new_value'
    var_13 = {var_11: var_12}
    var_14 = module_0.apply_overwrites_to_context(var_10, var_13)
    var_15 = 'choices'
    var_16 = 'a'
    var_17 = 'b'
    var_18 = 'c'
    var_19 = 'd'
    var_20 = [var_16, var_17, var_18, var_19]
    var_21 = {var_15: var_20}
    var_22 = [var_17, var_18]
    var_23 = {var_15: var_22}
    var_24 = module_0.apply_overwrites_to_context(var_21, var_23)
    var_25 = [var_16, var_17, var_18]
    var_26 = {var_15: var_25}
    var_27 = [var_17, var_19]
    var_28 = {var_15: var_27}
    var_29 = module_0.apply_overwrites_to_context(var_26, var_28)
    var_30 = 'choice'
    var_31 = 'default'
    var_32 = 'option1'
    var_33 = 'option2'
    var_34 = [var_31, var_32, var_33]
    var_35 = {var_30: var_34}
    var_36 = {var_30: var_33}
    var_37 = module_0.apply_overwrites_to_context(var_35, var_36)
    var_38 = [var_31, var_32]
    var_39 = {var_30: var_38}
    var_40 = 'invalid'
    var_41 = {var_30: var_40}
    var_42 = module_0.apply_overwrites_to_context(var_39, var_41)
    var_43 = 'config'
    var_44 = 'key1'
    var_45 = 'key2'
    var_46 = 'value1'
    var_47 = 'value2'
    var_48 = {var_44: var_46, var_45: var_47}
    var_49 = {var_43: var_48}
    var_50 = 'key3'
    var_51 = 'new_value2'
    var_52 = 'value3'
    var_53 = {var_45: var_51, var_50: var_52}
    var_54 = {var_43: var_53}
    var_55 = module_0.apply_overwrites_to_context(var_49, var_54)
    var_56 = 'flag'
    var_57 = True
    var_58 = {var_56: var_57}
    var_59 = 'no'
    var_60 = {var_56: var_59}
    var_61 = module_0.apply_overwrites_to_context(var_58, var_60)
    var_62 = False
    var_63 = {var_56: var_62}
    var_64 = 'YES'
    var_65 = {var_56: var_64}
    var_66 = module_0.apply_overwrites_to_context(var_63, var_65)
    var_67 = {var_56: var_57}
    var_68 = {var_56: var_40}
    var_69 = module_0.apply_overwrites_to_context(var_67, var_68)
    var_70 = 'nested'
    var_71 = 'inner'
    var_72 = {var_71: var_9}
    var_73 = {var_70: var_72}
    var_74 = 'new_key'
    var_75 = {var_74: var_12}
    var_76 = module_0.apply_overwrites_to_context(var_73, var_75, in_dictionary_variable=var_57)
    var_77 = 'data'
    var_78 = 'items'
    var_79 = [var_16, var_17, var_18]
    var_80 = {var_78: var_79}
    var_81 = {var_77: var_80}
    var_82 = 'x'
    var_83 = 'y'
    var_84 = [var_82, var_83]
    var_85 = {var_78: var_84}
    var_86 = {var_77: var_85}
    var_87 = module_0.apply_overwrites_to_context(var_81, var_86)
    var_88 = 'number'
    var_89 = 'text'
    var_90 = 42
    var_91 = 'hello'
    var_92 = {var_88: var_90, var_89: var_91}
    var_93 = 100
    var_94 = 'world'
    var_95 = {var_88: var_93, var_89: var_94}
    var_96 = module_0.apply_overwrites_to_context(var_92, var_95)
    var_97 = 'key'
    var_98 = {var_97: var_9}
    var_99 = {}
    var_100 = module_0.apply_overwrites_to_context(var_98, var_99)
    var_101 = {var_97: var_9}
    var_102 = None
    var_103 = {var_97: var_102}
    var_104 = module_0.apply_overwrites_to_context(var_101, var_103)



# Parsed testcases at query #4
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'cookiecutter'
    var_1 = '_copy_without_render'
    var_2 = '*.txt'
    var_3 = 'docs/*'
    var_4 = 'images/**'
    var_5 = [var_2, var_3, var_4]
    var_6 = {var_1: var_5}
    var_7 = {var_0: var_6}
    var_8 = 'readme.txt'
    var_9 = module_0.is_copy_only_path(var_8, var_7)
    assert var_9 is True
    var_10 = 'docs/index.md'
    var_11 = module_0.is_copy_only_path(var_10, var_7)
    assert var_11 is True
    var_12 = 'images/photo.jpg'
    var_13 = module_0.is_copy_only_path(var_12, var_7)
    assert var_13 is True
    var_14 = 'images/subfolder/photo.png'
    var_15 = module_0.is_copy_only_path(var_14, var_7)
    assert var_15 is True
    var_16 = 'src/main.py'
    var_17 = module_0.is_copy_only_path(var_16, var_7)
    assert var_17 is False
    var_18 = 'config.yaml'
    var_19 = module_0.is_copy_only_path(var_18, var_7)
    assert var_19 is False
    var_20 = {}
    var_21 = {var_0: var_20}
    var_22 = 'any/path'
    var_23 = module_0.is_copy_only_path(var_22, var_21)
    assert var_23 is False
    var_24 = []
    var_25 = {var_1: var_24}
    var_26 = {var_0: var_25}
    var_27 = module_0.is_copy_only_path(var_22, var_26)
    assert var_27 is False
    var_28 = 'exact_file.md'
    var_29 = [var_28]
    var_30 = {var_1: var_29}
    var_31 = {var_0: var_30}
    var_32 = module_0.is_copy_only_path(var_28, var_31)
    assert var_32 is True
    var_33 = 'other_file.md'
    var_34 = module_0.is_copy_only_path(var_33, var_31)
    assert var_34 is False
    var_35 = 'data/*.csv'
    var_36 = 'logs/*.log'
    var_37 = [var_35, var_36]
    var_38 = {var_1: var_37}
    var_39 = {var_0: var_38}
    var_40 = 'data/2023.csv'
    var_41 = module_0.is_copy_only_path(var_40, var_39)
    assert var_41 is True
    var_42 = 'logs/app.log'
    var_43 = module_0.is_copy_only_path(var_42, var_39)
    assert var_43 is True
    var_44 = 'data/backup/old.csv'
    var_45 = module_0.is_copy_only_path(var_44, var_39)
    assert var_45 is False
    var_46 = 'assets/**/*.png'
    var_47 = 'templates/*.html'
    var_48 = [var_46, var_47]
    var_49 = {var_1: var_48}
    var_50 = {var_0: var_49}
    var_51 = 'assets/images/icon.png'
    var_52 = module_0.is_copy_only_path(var_51, var_50)
    assert var_52 is True
    var_53 = 'assets/textures/wood/diffuse.png'
    var_54 = module_0.is_copy_only_path(var_53, var_50)
    assert var_54 is True
    var_55 = 'templates/index.html'
    var_56 = module_0.is_copy_only_path(var_55, var_50)
    assert var_56 is True
    var_57 = 'assets/readme.txt'
    var_58 = module_0.is_copy_only_path(var_57, var_50)
    assert var_58 is False



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0.0'
    var_9 = '2.0.0'
    var_10 = [var_8, var_9]
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = 'version'
    var_13 = '2.0.0'
    var_14 = {var_12: var_13}
    var_15 = 'project_name'
    var_16 = 'open_source'
    var_17 = 'Test'
    var_18 = True
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'project_name'
    var_21 = 'Overridden'
    var_22 = {var_20: var_21}
    var_23 = '{invalid json'
    var_24 = 'project'
    var_25 = 'features'
    var_26 = 'name'
    var_27 = 'settings'
    var_28 = 'Test'
    var_29 = 'debug'
    var_30 = True
    var_31 = {var_29: var_30}
    var_32 = {var_26: var_28, var_27: var_31}
    var_33 = 'auth'
    var_34 = 'api'
    var_35 = [var_33, var_34]
    var_36 = {var_24: var_32, var_25: var_35}
    var_37 = 'name'
    var_38 = 'version'
    var_39 = 'settings'
    var_40 = 'Default Name'
    var_41 = '1.0'
    var_42 = '2.0'
    var_43 = [var_41, var_42]
    var_44 = 'debug'
    var_45 = False
    var_46 = {var_44: var_45}
    var_47 = {var_37: var_40, var_38: var_43, var_39: var_46}
    var_48 = 'version'
    var_49 = '2.0'
    var_50 = {var_48: var_49}
    var_51 = 'name'
    var_52 = 'settings'
    var_53 = 'Final Name'
    var_54 = 'debug'
    var_55 = True
    var_56 = {var_54: var_55}
    var_57 = {var_51: var_53, var_52: var_56}



# Parsed testcases at query #6
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'project_name'
    var_11 = 'Default Project'
    var_12 = {var_10: var_11}
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'Test'
    var_16 = '1.0.0'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'project_name'
    var_19 = 'Extra Project'
    var_20 = {var_18: var_19}
    var_21 = '{"invalid": json}'
    var_22 = 'project'
    var_23 = 'list'
    var_24 = 'name'
    var_25 = 'settings'
    var_26 = 'Test'
    var_27 = 'debug'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = {var_24: var_26, var_25: var_29}
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 'c'
    var_34 = [var_31, var_32, var_33]
    var_35 = {var_22: var_30, var_23: var_34}
    var_36 = 'name'
    var_37 = 'version'
    var_38 = 'Original'
    var_39 = '1.0'
    var_40 = {var_36: var_38, var_37: var_39}
    var_41 = 'name'
    var_42 = 'Default'
    var_43 = {var_41: var_42}
    var_44 = 'version'
    var_45 = '2.0'
    var_46 = {var_44: var_45}



# Parsed testcases at query #7
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'project_name'
    var_11 = 'Default Project'
    var_12 = {var_10: var_11}
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'Test'
    var_16 = '1.0'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'project_name'
    var_19 = 'Extra Project'
    var_20 = {var_18: var_19}
    var_21 = 'project_name'
    var_22 = 'version'
    var_23 = 'Original'
    var_24 = '1.0'
    var_25 = {var_21: var_23, var_22: var_24}
    var_26 = 'project_name'
    var_27 = 'Default Project'
    var_28 = {var_26: var_27}
    var_29 = 'Extra Project'
    var_30 = {var_26: var_29}
    var_31 = '{invalid json'
    var_32 = 'project'
    var_33 = 'list'
    var_34 = 'name'
    var_35 = 'settings'
    var_36 = 'Test'
    var_37 = 'debug'
    var_38 = True
    var_39 = {var_37: var_38}
    var_40 = {var_34: var_36, var_35: var_39}
    var_41 = 'a'
    var_42 = 'b'
    var_43 = 'c'
    var_44 = [var_41, var_42, var_43]
    var_45 = {var_32: var_40, var_33: var_44}
    var_46 = 'test'
    var_47 = 'value'
    var_48 = {var_46: var_47}
    var_49 = 0
    var_50 = 1
    var_51 = '.'
    var_52 = var_2.split(var_51)[var_49]



# Parsed testcases at query #8
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'project_name'
    var_1 = 'MyProject'
    var_2 = {var_0: var_1}
    var_3 = module_0.Environment()
    var_4 = '{{ project_name }}_app'
    var_5 = False
    var_6 = 'MyProject_app'
    var_7 = '{{ project_name }}_app'
    var_8 = False
    var_9 = True
    var_10 = ''
    var_11 = False
    var_12 = 'user'
    var_13 = 'version'
    var_14 = 'type'
    var_15 = 'john'
    var_16 = '1.0'
    var_17 = 'backend'
    var_18 = {var_12: var_15, var_13: var_16, var_14: var_17}
    var_19 = '{{ user }}-{{ type }}-v{{ version }}'
    var_20 = 'john-backend-v1.0'
    var_21 = 'projects/{{ user }}/src'
    var_22 = 'projects'
    var_23 = 'src'
    var_24 = 'name'
    var_25 = 'test-project_123'
    var_26 = {var_24: var_25}
    var_27 = '{{ name }}'
    var_28 = 'path_object_test'
    var_29 = {}
    var_30 = 'string_output_test'
    var_31 = {}



# Parsed testcases at query #9
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = True
    var_2 = module_0.Environment()
    var_3 = 'project_name'
    var_4 = 'MyProject'
    var_5 = {var_3: var_4}
    var_6 = '{{ project_name }}_dir'
    var_7 = 'ExistingDir'
    var_8 = 'ExistingDir'
    var_9 = {}
    var_10 = False
    var_11 = {}
    var_12 = ''
    var_13 = {}
    var_14 = 'user'
    var_15 = 'version'
    var_16 = 'testuser'
    var_17 = '1.0'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = 'app_{{ user }}_{{ version }}'
    var_20 = 'src/{{ project_name }}/tests'
    var_21 = 'MyApp'
    var_22 = {var_3: var_21}
    var_23 = 'src'
    var_24 = 'tests'
    var_25 = 'name'
    var_26 = 'test-project'
    var_27 = {var_25: var_26}
    var_28 = '{{ name }}_folder'



# Parsed testcases at query #10
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0.0'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'project_name'
    var_11 = 'Default Project'
    var_12 = {var_10: var_11}
    var_13 = 'project_name'
    var_14 = 'version'
    var_15 = 'Test'
    var_16 = '1.0.0'
    var_17 = {var_13: var_15, var_14: var_16}
    var_18 = 'project_name'
    var_19 = 'Overridden Project'
    var_20 = {var_18: var_19}
    var_21 = '{"invalid": json}'
    var_22 = 'project'
    var_23 = 'list'
    var_24 = 'name'
    var_25 = 'settings'
    var_26 = 'Test'
    var_27 = 'debug'
    var_28 = True
    var_29 = {var_27: var_28}
    var_30 = {var_24: var_26, var_25: var_29}
    var_31 = 'a'
    var_32 = 'b'
    var_33 = 'c'
    var_34 = [var_31, var_32, var_33]
    var_35 = {var_22: var_30, var_23: var_34}
    var_36 = 'key'
    var_37 = 'value'
    var_38 = {var_36: var_37}
    var_39 = var_36.stem



# Parsed testcases at query #11
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = module_0.Environment()
    var_1 = 'project_name'
    var_2 = 'test_project'
    var_3 = {var_1: var_2}
    var_4 = ''
    var_5 = module_0.Environment()
    var_6 = 'project_name'
    var_7 = 'test_project'
    var_8 = {var_6: var_7}
    var_9 = '   '
    var_10 = module_0.Environment()
    var_11 = 'project_name'
    var_12 = 'my_project'
    var_13 = {var_11: var_12}
    var_14 = '{{ project_name }}_dir'
    var_15 = 'my_project_dir'
    var_16 = module_0.Environment()
    var_17 = 'project_name'
    var_18 = 'existing_project'
    var_19 = {var_17: var_18}
    var_20 = 'existing_project_dir'
    var_21 = '{{ project_name }}_dir'
    var_22 = False
    var_23 = module_0.Environment()
    var_24 = 'project_name'
    var_25 = 'overwritten_project'
    var_26 = {var_24: var_25}
    var_27 = 'overwritten_project_dir'
    var_28 = '{{ project_name }}_dir'
    var_29 = True
    var_30 = module_0.Environment()
    var_31 = 'project_name'
    var_32 = 'version'
    var_33 = 'author'
    var_34 = 'complex'
    var_35 = '1.0'
    var_36 = 'test_author'
    var_37 = {var_31: var_34, var_32: var_35, var_33: var_36}
    var_38 = '{{ project_name }}-v{{ version }}-by-{{ author }}'
    var_39 = 'complex-v1.0-by-test_author'
    var_40 = module_0.Environment()
    var_41 = 'project_name'
    var_42 = 'nested_project'
    var_43 = {var_41: var_42}
    var_44 = 'projects/{{ project_name }}/src'
    var_45 = 'projects'
    var_46 = var_34 / var_42
    var_47 = 'src'
    var_48 = var_46 / var_47
    var_49 = module_0.Environment()
    var_50 = 'name'
    var_51 = 'test@project#123'
    var_52 = {var_50: var_51}
    var_53 = '{{ name }}'
    var_54 = module_0.Environment()
    var_55 = 'project_name'
    var_56 = 'string_path'
    var_57 = {var_55: var_56}
    var_58 = '{{ project_name }}_test'
    var_59 = 'string_path_test'



# Parsed testcases at query #12
#--------------------------


def test_case_0():
    var_0 = 'project_name'
    var_1 = 'version'
    var_2 = 'Test Project'
    var_3 = '1.0.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'project_name'
    var_6 = 'version'
    var_7 = 'Test'
    var_8 = '1.0.0'
    var_9 = '2.0.0'
    var_10 = [var_8, var_9]
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = 'version'
    var_13 = '2.0.0'
    var_14 = {var_12: var_13}
    var_15 = 'project_name'
    var_16 = 'open_source'
    var_17 = 'Test'
    var_18 = True
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = 'project_name'
    var_21 = 'open_source'
    var_22 = 'Overridden'
    var_23 = 'no'
    var_24 = {var_20: var_22, var_21: var_23}
    var_25 = '{"invalid": json}'
    var_26 = 'project'
    var_27 = 'choices'
    var_28 = 'name'
    var_29 = 'config'
    var_30 = 'Test'
    var_31 = 'debug'
    var_32 = True
    var_33 = {var_31: var_32}
    var_34 = {var_28: var_30, var_29: var_33}
    var_35 = 'opt1'
    var_36 = 'opt2'
    var_37 = 'opt3'
    var_38 = [var_35, var_36, var_37]
    var_39 = {var_26: var_34, var_27: var_38}
    var_40 = 'project'
    var_41 = 'choices'
    var_42 = 'config'
    var_43 = 'debug'
    var_44 = False
    var_45 = {var_43: var_44}
    var_46 = {var_42: var_45}
    var_47 = 'opt2'
    var_48 = {var_40: var_46, var_41: var_47}
    var_49 = 'choice'
    var_50 = 'a'
    var_51 = 'b'
    assert var_51 == 1
    var_52 = 'c'
    var_53 = [var_50, var_51, var_52]
    var_54 = {var_49: var_53}
    var_55 = 'choice'
    var_56 = 'invalid'
    var_57 = {var_55: var_56}
    var_58 = 'always'
    var_59 = 0
    var_60 = var_53.message
    var_61 = str(var_60)



# Parsed testcases at query #13
#--------------------------


import jinja2.environment as module_0

def test_case_0():
    var_0 = 'output'
    var_1 = 'cookiecutter'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'TestProject'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_1: var_6}
    var_8 = module_0.Environment()
    var_9 = '{{ cookiecutter.project_name }}_{{ cookiecutter.version }}'
    var_10 = 'TestProject_1.0.0'
    var_11 = True
    var_12 = ''
    var_13 = 'projects/{{ cookiecutter.project_name }}/src'
    var_14 = 'projects/TestProject/src'
    var_15 = '{{ cookiecutter.undefined_var }}'
    var_16 = 'simple'
    var_17 = 'cookiecutter'
    var_18 = 'name'
    var_19 = 'Test@Project#123'
    var_20 = {var_18: var_19}
    var_21 = {var_17: var_20}
    var_22 = '{{ cookiecutter.name }}'
    var_23 = '../outside_dir'
    var_24 = '../outside_dir'
    var_25 = 'a/b/c/{{ cookiecutter.project_name }}'
    var_26 = 'a/b/c/TestProject'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'template'
    var_1 = 'project'
    var_2 = 'test.txt.j2'
    var_3 = 'Hello {{ name }}!'
    var_4 = 'binary.bin'
    var_5 = b'\x00\x01\x02\x03'
    var_6 = 'name'
    var_7 = 'cookiecutter'
    var_8 = 'World'
    var_9 = '_new_lines'
    var_10 = '\n'
    var_11 = {var_9: var_10}
    var_12 = {var_6: var_8, var_7: var_11}
    var_13 = 'test.txt'
    var_14 = {}
    var_15 = {var_7: var_14}
    var_16 = 'existing.txt'
    var_17 = 'Original content'
    var_18 = 'existing.txt.j2'
    var_19 = 'New content'
    var_20 = True
    var_21 = 'empty_dir'
    var_22 = '.keep'
    var_23 = 'content'
    var_24 = 'empty_dir/.keep'
    var_25 = 'bad.txt.j2'
    var_26 = 'Hello {{ name }'
    var_27 = 'bad.txt.j2'
    var_28 = 'undefined.txt.j2'
    var_29 = 'Hello {{ undefined_var }}!'
    var_30 = 'undefined.txt.j2'
    var_31 = 'newline.txt.j2'
    var_32 = 'Line1\nLine2\r\nLine3'
    var_33 = '\r\n'
    var_34 = {var_9: var_33}
    var_35 = {var_7: var_34}
    var_36 = 'newline.txt'
    var_37 = 'test.txt.j2'



# Parsed testcases at query #15
#--------------------------


import json as module_0
import cookiecutter.generate as module_1

def test_case_0():
    var_0 = 'template'
    var_1 = 'cookiecutter.json'
    var_2 = 'project_name'
    var_3 = 'version'
    var_4 = 'Test Project'
    var_5 = '1.0.0'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.dumps(var_6)
    var_8 = '{{ cookiecutter.project_name }}.txt'
    var_9 = 'Version: {{ cookiecutter.version }}'
    var_10 = 'output'
    var_11 = False
    var_12 = 'Test Project.txt'
    var_13 = 'template'
    var_14 = 'cookiecutter.json'
    var_15 = 'name'
    var_16 = 'Test'
    var_17 = {var_15: var_16}
    var_18 = module_0.dumps(var_17)
    var_19 = 'test.txt'
    var_20 = 'content'
    var_21 = 'output'
    var_22 = var_9 / var_21
    var_23 = var_22 / var_16
    var_24 = str(var_22)
    var_25 = False
    var_26 = str(var_22)
    var_27 = True
    var_28 = 'template'
    var_29 = 'cookiecutter.json'
    var_30 = 'name'
    var_31 = 'Test'
    var_32 = {var_30: var_31}
    var_33 = module_0.dumps(var_32)
    var_34 = 'test.txt'
    var_35 = 'new content'
    var_36 = 'output'
    var_37 = var_9 / var_36
    var_38 = var_37 / var_31
    var_39 = var_38 / var_34
    var_40 = 'old content'
    var_41 = str(var_37)
    var_42 = True
    var_43 = module_1.generate_files(var_11, output_dir=var_41, overwrite_if_exists=var_42, skip_if_file_exists=var_42)
    var_44 = 'template'
    var_45 = 'cookiecutter.json'
    var_46 = 'defined_var'
    var_47 = 'value'
    var_48 = {var_46: var_47}
    var_49 = module_0.dumps(var_48)
    var_50 = '{{ undefined_var }}.txt'
    var_51 = 'content'
    var_52 = 'output'
    var_53 = var_9 / var_52
    var_54 = str(var_53)
    var_55 = 'template'
    var_56 = 'cookiecutter.json'
    var_57 = 'project_name'
    var_58 = '_copy_without_render'
    var_59 = 'MyProject'
    var_60 = 'static/*'
    var_61 = 'config.json'
    var_62 = [var_60, var_61]
    var_63 = {var_57: var_59, var_58: var_62}
    var_64 = module_0.dumps(var_63)
    var_65 = 'static'
    var_66 = 'image.png'
    var_67 = b'binary content'
    var_68 = '{"key": "{{ cookiecutter.project_name }}"}'
    var_69 = 'README.md'
    var_70 = '# {{ cookiecutter.project_name }}'
    var_71 = 'output'
    var_72 = str(var_53)
    var_73 = 'template'
    var_74 = 'cookiecutter.json'
    var_75 = 'project_name'
    var_76 = 'version'
    var_77 = 'Default'
    var_78 = '1.0.0'
    var_79 = {var_75: var_77, var_76: var_78}
    var_80 = module_0.dumps(var_79)
    var_81 = '{{ cookiecutter.project_name }}.txt'
    var_82 = 'Version: {{ cookiecutter.version }}'
    var_83 = 'output'
    var_84 = str(var_53)
    var_85 = 'Custom'
    assert var_85 == b'\x00\x01\x02\x03\x04'
    var_86 = {var_75: var_85}
    var_87 = module_1.generate_files(var_40, output_dir=var_84)
    var_88 = 'Custom.txt'
    var_89 = var_53 / var_88
    var_90 = 'template'
    var_91 = 'cookiecutter.json'
    var_92 = 'name'
    var_93 = 'Test'
    var_94 = {var_92: var_93}
    var_95 = module_0.dumps(var_94)
    var_96 = 'binary.dat'
    var_97 = b'\x00\x01\x02\x03\x04'
    var_98 = 'output'
    var_99 = var_82 / var_98
    var_100 = str(var_99)
    var_101 = module_1.generate_files(var_83, output_dir=var_100)
    var_102 = var_40 / var_96
    var_103 = 'template'
    var_104 = 'cookiecutter.json'
    var_105 = 'project_name'
    var_106 = 'module_name'
    var_107 = 'NestedTest'
    var_108 = 'mymodule'
    var_109 = {var_105: var_107, var_106: var_108}
    var_110 = module_0.dumps(var_109)
    var_111 = 'src'
    var_112 = '{{ cookiecutter.module_name }}'
    var_113 = var_82 / var_112
    var_114 = True
    var_115 = '__init__.py'
    var_116 = var_113 / var_115
    var_117 = '# {{ cookiecutter.project_name }}'
    var_118 = 'main.py'
    var_119 = var_113 / var_118
    var_120 = "print('{{ cookiecutter.project_name }}')"
    var_121 = 'output'
    var_122 = str(var_99)
    var_123 = var_72 / var_111
    var_124 = var_123 / var_108
    var_125 = var_124 / var_115



# Parsed testcases at query #16
#--------------------------


import cookiecutter.generate as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'version'
    var_2 = 'old'
    var_3 = '1.0'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'new'
    var_6 = {var_0: var_5}
    var_7 = module_0.apply_overwrites_to_context(var_4, var_6)
    var_8 = {var_0: var_2}
    var_9 = 'new_var'
    var_10 = 'value'
    var_11 = {var_9: var_10}
    var_12 = module_0.apply_overwrites_to_context(var_8, var_11)
    var_13 = 'choices'
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'c'
    var_17 = [var_14, var_15, var_16]
    var_18 = {var_13: var_17}
    var_19 = {var_13: var_15}
    var_20 = module_0.apply_overwrites_to_context(var_18, var_19)
    var_21 = [var_14, var_15, var_16]
    var_22 = {var_13: var_21}
    var_23 = 'd'
    var_24 = {var_13: var_23}
    var_25 = module_0.apply_overwrites_to_context(var_22, var_24)
    var_26 = 'multichoice'
    var_27 = [var_14, var_15, var_16, var_23]
    var_28 = {var_26: var_27}
    var_29 = [var_15, var_16]
    var_30 = {var_26: var_29}
    var_31 = module_0.apply_overwrites_to_context(var_28, var_30)
    var_32 = [var_14, var_15, var_16]
    var_33 = {var_26: var_32}
    var_34 = [var_15, var_23]
    var_35 = {var_26: var_34}
    var_36 = module_0.apply_overwrites_to_context(var_33, var_35)
    var_37 = 'config'
    var_38 = 'enabled'
    var_39 = True
    var_40 = {var_36: var_2, var_1: var_3, var_38: var_39}
    var_41 = {var_37: var_40}
    var_42 = 'extra'
    var_43 = {var_36: var_5, var_42: var_10}
    var_44 = {var_37: var_43}
    var_45 = module_0.apply_overwrites_to_context(var_41, var_44)
    var_46 = False
    var_47 = {var_38: var_46}
    var_48 = 'yes'
    var_49 = {var_38: var_48}
    var_50 = module_0.apply_overwrites_to_context(var_47, var_49)
    var_51 = {var_38: var_39}
    var_52 = 'no'
    var_53 = {var_38: var_52}
    var_54 = module_0.apply_overwrites_to_context(var_51, var_53)
    var_55 = {var_38: var_39}
    var_56 = 'maybe'
    var_57 = {var_38: var_56}
    var_58 = module_0.apply_overwrites_to_context(var_55, var_57)
    var_59 = 'nested'
    var_60 = 'items'
    var_61 = [var_14, var_15]
    var_62 = {var_60: var_61}
    var_63 = {var_59: var_62}
    var_64 = [var_16, var_23]
    var_65 = {var_60: var_64}
    var_66 = {var_59: var_65}
    var_67 = module_0.apply_overwrites_to_context(var_63, var_66, in_dictionary_variable=var_39)
    var_68 = 'project'
    var_69 = 'settings'
    var_70 = 'features'
    var_71 = 'test'
    var_72 = 'debug'
    var_73 = 'log_level'
    var_74 = 'info'
    var_75 = {var_72: var_46, var_73: var_74}
    var_76 = 'auth'
    var_77 = 'api'
    var_78 = [var_76, var_77]
    var_79 = {var_58: var_71, var_69: var_75, var_70: var_78}
    var_80 = {var_68: var_79}
    var_81 = 'updated'
    var_82 = 'new_setting'
    var_83 = {var_72: var_48, var_82: var_10}
    var_84 = [var_77]
    var_85 = {var_58: var_81, var_69: var_83, var_70: var_84}
    var_86 = {var_68: var_85}
    var_87 = module_0.apply_overwrites_to_context(var_80, var_86)



