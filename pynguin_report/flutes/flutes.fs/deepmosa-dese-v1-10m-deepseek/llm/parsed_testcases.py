####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_cache_saves_and_loads_from_file. Retrieved 1/15 statements.
# Partially parsed test_cache_does_not_save_when_path_is_none. Retrieved 3/7 statements.
# Partially parsed test_cache_loads_from_existing_file. Retrieved 2/14 statements.
# Partially parsed test_cache_logs_when_verbose_is_true. Retrieved 2/12 statements.


def test_case_0():
    var_0 = False

import flutes.fs as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.cache(var_0, var_1)

def test_case_0():
    var_0 = 42
    var_1 = False

def test_case_0():
    var_0 = True
    var_1 = 'test'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_copy_tree_creates_destination_directory. Retrieved 4/6 statements.
# Partially parsed test_copy_tree_copies_files. Retrieved 6/11 statements.
# Partially parsed test_copy_tree_overwrites_files_when_overwrite_is_true. Retrieved 6/14 statements.
# Partially parsed test_copy_tree_does_not_overwrite_files_when_overwrite_is_false. Retrieved 7/15 statements.
# Partially parsed test_copy_tree_copies_subdirectories. Retrieved 5/10 statements.


import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/source_dir'
    var_1 = '/tmp/destination_dir'
    var_2 = True
    var_3 = module_0.copy_tree(var_0, var_1)

import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/source_dir'
    var_1 = '/tmp/destination_dir'
    var_2 = True
    var_3 = 'test content'
    var_4 = module_0.copy_tree(var_0, var_1)
    var_5 = 'test_file.txt'

import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/source_dir'
    var_1 = '/tmp/destination_dir'
    var_2 = True
    var_3 = 'new content'
    var_4 = 'old content'
    assert var_4 == 'new content'
    var_5 = module_0.copy_tree(var_0, var_1, var_4)

import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/source_dir'
    var_1 = '/tmp/destination_dir'
    var_2 = True
    var_3 = 'new content'
    var_4 = 'old content'
    assert var_4 == 'old content'
    var_5 = False
    var_6 = module_0.copy_tree(var_0, var_1, var_5)

import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/source_dir'
    var_1 = '/tmp/destination_dir'
    var_2 = True
    var_3 = 'subdir'
    var_4 = module_0.copy_tree(var_0, var_1)



# Parsed testcases at query #3
#--------------------------




import flutes.fs as module_0

def test_case_0():
    var_0 = 500
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '500.00'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '1.00M'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00G'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '1.00T'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00P'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = 0
    var_2 = module_0.readable_size(var_0, var_1)
    assert var_2 == '1K'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1500
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.46K'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1024.00P'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_copy_tree_creates_destination_directory. Retrieved 1/12 statements.
# Partially parsed test_copy_tree_copies_files. Retrieved 2/14 statements.
# Partially parsed test_copy_tree_overwrites_files_when_overwrite_is_true. Retrieved 3/17 statements.
# Partially parsed test_copy_tree_does_not_overwrite_files_when_overwrite_is_false. Retrieved 3/17 statements.
# Partially parsed test_copy_tree_copies_subdirectories. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'new_dir'

def test_case_0():
    var_0 = 'test'
    var_1 = 'test_file.txt'

def test_case_0():
    var_0 = 'test'
    var_1 = 'old_content'
    var_2 = True
    assert var_2 == 'test'

def test_case_0():
    var_0 = 'test'
    var_1 = 'old_content'
    var_2 = False
    assert var_2 == 'old_content'

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'test'
    var_2 = 'test_file.txt'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_cache_no_path. Retrieved 1/2 statements.
# Partially parsed test_cache_with_path. Retrieved 1/2 statements.
# Partially parsed test_cache_load_from_existing_file. Retrieved 1/2 statements.
# Partially parsed test_cache_verbose_false. Retrieved 1/2 statements.
# Partially parsed test_cache_custom_name. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 42
    assert var_0 == 42

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 42



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_scandir_with_pathlib_path. Retrieved 3/11 statements.
# Partially parsed test_scandir_with_str_path. Retrieved 8/10 statements.
# Partially parsed test_scandir_empty_directory_with_pathlib_path. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_directory'
    var_1 = 0
    var_2 = 1

import flutes.fs as module_0

def test_case_0():
    var_0 = 'test_directory'
    var_1 = module_0.scandir(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6]

def test_case_0():
    var_0 = 'empty_directory'

import flutes.fs as module_0

def test_case_0():
    var_0 = 'empty_directory'
    var_1 = module_0.scandir(var_0)
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 0



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_scandir_yields_string_path_when_input_is_string. Retrieved 1/12 statements.
# Partially parsed test_scandir_yields_path_object_when_input_is_path. Retrieved 1/13 statements.


def test_case_0():
    var_0 = 'test.txt'

def test_case_0():
    var_0 = 'test.txt'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_scandir_path_type_check. Retrieved 1/5 statements.


def test_case_0():
    var_0 = '/some/directory'



# Parsed testcases at query #9
#--------------------------




import flutes.fs as module_0

def test_case_0():
    var_0 = '/non/existent/directory'
    var_1 = module_0.scandir(var_0)
    var_2 = list(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_copy_tree_overwrite. Retrieved 7/17 statements.


import flutes.fs as module_0

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'test_dst'
    var_2 = 'file.txt'
    var_3 = 'source content'
    var_4 = 'destination content'
    var_5 = True
    var_6 = module_0.copy_tree(var_0, var_1, var_5)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_copy_tree_overwrite_true. Retrieved 4/5 statements.
# Partially parsed test_copy_tree_overwrite_false. Retrieved 4/5 statements.


import flutes.fs as module_0

def test_case_0():
    var_0 = '/path/to/source'
    var_1 = '/path/to/destination'
    var_2 = True
    var_3 = module_0.copy_tree(var_0, var_1, var_2)

import flutes.fs as module_0

def test_case_0():
    var_0 = '/path/to/source'
    var_1 = '/path/to/destination'
    var_2 = False
    var_3 = module_0.copy_tree(var_0, var_1, var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_scandir_with_str_path. Retrieved 3/16 statements.
# Partially parsed test_scandir_with_pathlib_path. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'
    var_2 = 0

def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'
    var_2 = 0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_cache_with_path_and_verbose_true. Retrieved 2/17 statements.
# Partially parsed test_cache_with_path_and_verbose_false. Retrieved 2/17 statements.
# Partially parsed test_cache_with_path_none. Retrieved 4/9 statements.
# Partially parsed test_cache_with_existing_cache_file. Retrieved 3/19 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test'

def test_case_0():
    var_0 = False
    var_1 = 'test'

import flutes.fs as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'test'
    var_3 = module_0.cache(var_0, var_1, var_2)

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = 'test'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_copy_tree_overwrite_true. Retrieved 6/14 statements.


import flutes.fs as module_0

def test_case_0():
    var_0 = '/tmp/test_src'
    var_1 = '/tmp/test_dst'
    var_2 = True
    var_3 = 'src content'
    var_4 = 'dst content'
    var_5 = module_0.copy_tree(var_0, var_1, var_4)



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_scandir_with_path_object.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_scandir_with_str_path. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test.txt'
    var_1 = 'test'
    var_2 = 0



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_scandir_with_pathlib_path. Retrieved 4/27 statements.
# Partially parsed test_scandir_with_str_path. Retrieved 5/27 statements.


def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'

def test_case_0():
    var_0 = 'test_dir'
    var_1 = True
    var_2 = 'file1.txt'
    var_3 = 'file2.txt'
    var_4 = 'w'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_copy_tree_overwrite_true_evaluates_to_true. Retrieved 7/17 statements.
# Partially parsed test_copy_tree_file_not_exists_evaluates_to_true. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'test_src'
    var_1 = 'test_dst'
    var_2 = True
    var_3 = 'test'
    var_4 = 'old'
    var_5 = True
    var_6 = 'test_file'

def test_case_0():
    var_0 = 'test_src'
    var_1 = 'test_dst'
    var_2 = True
    var_3 = 'test'
    var_4 = False
    var_5 = 'test_file'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_cache_with_file_and_verbose. Retrieved 3/17 statements.
# Partially parsed test_cache_with_file_and_no_verbose. Retrieved 3/17 statements.
# Partially parsed test_cache_with_no_file. Retrieved 2/15 statements.
# Partially parsed test_cache_with_no_path. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = 'test_cache'

def test_case_0():
    var_0 = 123
    var_1 = False
    var_2 = 'test_cache'

def test_case_0():
    var_0 = True
    var_1 = 'test_cache'

import flutes.fs as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'test_cache'
    var_3 = module_0.cache(var_0, var_1, var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_copy_tree_overwrite_false_destination_exists. Retrieved 3/6 statements.


def test_case_0():
    var_0 = '/path/to/source'
    var_1 = '/path/to/destination'
    var_2 = False



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import flutes.fs as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '0.00'
    var_2 = 500
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '500.00'
    var_4 = 1023
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1023.00'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = 2048
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '2.00K'
    var_4 = 1536
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.50K'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '1.00M'
    var_3 = 2.5
    var_4 = var_3 * var_0
    var_5 = var_4 * var_0
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '2.50M'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00G'
    var_4 = 3.75
    var_5 = var_4 * var_0
    var_6 = var_5 * var_0
    var_7 = var_6 * var_0
    var_8 = module_0.readable_size(var_7)
    assert var_8 == '3.75G'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '1.00T'
    var_5 = 5.25
    var_6 = var_5 * var_0
    var_7 = var_6 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '5.25T'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00P'
    var_6 = 7.125
    var_7 = var_6 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = var_9 * var_0
    var_11 = var_10 * var_0
    var_12 = module_0.readable_size(var_11)
    assert var_12 == '7.12P'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = 0
    var_2 = module_0.readable_size(var_0, var_1)
    assert var_2 == '1K'
    var_3 = 1536
    var_4 = 1
    var_5 = module_0.readable_size(var_3, var_4)
    assert var_5 == '1.5K'
    var_6 = 2048
    var_7 = 3
    var_8 = module_0.readable_size(var_6, var_7)
    assert var_8 == '2.000K'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_cache_with_path_and_verbose. Retrieved 2/14 statements.
# Partially parsed test_cache_with_path_and_no_verbose. Retrieved 2/14 statements.
# Partially parsed test_cache_with_no_path. Retrieved 4/14 statements.
# Partially parsed test_cache_with_name. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = True

def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = False

import flutes.fs as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.cache(var_0, var_1)
    var_3 = 'cache.pkl'

def test_case_0():
    var_0 = 'cache.pkl'
    var_1 = True
    var_2 = 'test'



# Parsed testcases at query #2
#--------------------------




import flutes.fs as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '0.00'
    var_2 = 1
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00'
    var_4 = 1023
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1023.00'
    var_6 = module_0.readable_size(var_4, var_0)
    assert var_6 == '1023'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = module_0.readable_size(var_0)
    assert var_1 == '1.00K'
    var_2 = 1.5
    var_3 = var_0 * var_2
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '1.50K'
    var_5 = var_0 * var_0
    var_6 = 1
    var_7 = var_5 - var_6
    var_8 = module_0.readable_size(var_7)
    assert var_8 == '1024.00K'
    var_9 = var_0 * var_0
    var_10 = var_9 - var_6
    var_11 = 0
    var_12 = module_0.readable_size(var_10, var_11)
    assert var_12 == '1024K'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = module_0.readable_size(var_1)
    assert var_2 == '1.00M'
    var_3 = var_0 * var_0
    var_4 = 1.5
    var_5 = var_3 * var_4
    var_6 = module_0.readable_size(var_5)
    assert var_6 == '1.50M'
    var_7 = var_0 * var_0
    var_8 = var_7 * var_0
    var_9 = 1
    var_10 = var_8 - var_9
    var_11 = module_0.readable_size(var_10)
    assert var_11 == '1024.00M'
    var_12 = var_0 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 - var_9
    var_15 = 0
    var_16 = module_0.readable_size(var_14, var_15)
    assert var_16 == '1024M'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = module_0.readable_size(var_2)
    assert var_3 == '1.00G'
    var_4 = var_0 * var_0
    var_5 = var_4 * var_0
    var_6 = 1.5
    var_7 = var_5 * var_6
    var_8 = module_0.readable_size(var_7)
    assert var_8 == '1.50G'
    var_9 = var_0 * var_0
    var_10 = var_9 * var_0
    var_11 = var_10 * var_0
    var_12 = 1
    var_13 = var_11 - var_12
    var_14 = module_0.readable_size(var_13)
    assert var_14 == '1024.00G'
    var_15 = var_0 * var_0
    var_16 = var_15 * var_0
    var_17 = var_16 * var_0
    var_18 = var_17 - var_12
    var_19 = 0
    var_20 = module_0.readable_size(var_18, var_19)
    assert var_20 == '1024G'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = module_0.readable_size(var_3)
    assert var_4 == '1.00T'
    var_5 = var_0 * var_0
    var_6 = var_5 * var_0
    var_7 = var_6 * var_0
    var_8 = 1.5
    var_9 = var_7 * var_8
    var_10 = module_0.readable_size(var_9)
    assert var_10 == '1.50T'
    var_11 = var_0 * var_0
    var_12 = var_11 * var_0
    var_13 = var_12 * var_0
    var_14 = var_13 * var_0
    var_15 = 1
    var_16 = var_14 - var_15
    var_17 = module_0.readable_size(var_16)
    assert var_17 == '1024.00T'
    var_18 = var_0 * var_0
    var_19 = var_18 * var_0
    var_20 = var_19 * var_0
    var_21 = var_20 * var_0
    var_22 = var_21 - var_15
    var_23 = 0
    var_24 = module_0.readable_size(var_22, var_23)
    assert var_24 == '1024T'

import flutes.fs as module_0

def test_case_0():
    var_0 = 1024
    var_1 = var_0 * var_0
    var_2 = var_1 * var_0
    var_3 = var_2 * var_0
    var_4 = var_3 * var_0
    var_5 = module_0.readable_size(var_4)
    assert var_5 == '1.00P'
    var_6 = var_0 * var_0
    var_7 = var_6 * var_0
    var_8 = var_7 * var_0
    var_9 = var_8 * var_0
    var_10 = 1.5
    var_11 = var_9 * var_10
    var_12 = module_0.readable_size(var_11)
    assert var_12 == '1.50P'
    var_13 = var_0 * var_0
    var_14 = var_13 * var_0
    var_15 = var_14 * var_0
    var_16 = var_15 * var_0
    var_17 = var_16 * var_0
    var_18 = module_0.readable_size(var_17)
    assert var_18 == '1024.00P'
    var_19 = var_0 * var_0
    var_20 = var_19 * var_0
    var_21 = var_20 * var_0
    var_22 = var_21 * var_0
    var_23 = var_22 * var_0
    var_24 = 0
    var_25 = module_0.readable_size(var_23, var_24)
    assert var_25 == '1024P'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_scandir_with_pathlib_path. Retrieved 2/22 statements.
# Partially parsed test_scandir_with_str_path. Retrieved 3/21 statements.


def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'file2.txt'

def test_case_0():
    var_0 = 'file1.txt'
    var_1 = 'w'
    var_2 = 'file2.txt'



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_cache_file_exists.




# Parsed testcases at query #5
#--------------------------

# Partially parsed test_copy_tree_new_directory. Retrieved 4/24 statements.
# Partially parsed test_copy_tree_overwrite_existing_files. Retrieved 7/37 statements.
# Partially parsed test_copy_tree_skip_existing_files. Retrieved 7/37 statements.
# Failed to parse test_copy_tree_empty_directory.


def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'w'
    var_3 = 'file2.txt'

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'w'
    var_3 = 'src content'
    var_4 = 'file2.txt'
    var_5 = 'dst content'
    var_6 = True

def test_case_0():
    var_0 = 'subdir'
    var_1 = 'file1.txt'
    var_2 = 'w'
    var_3 = 'src content'
    var_4 = 'file2.txt'
    var_5 = 'dst content'
    var_6 = False



