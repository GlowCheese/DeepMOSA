####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_text_constructor_default_locale. Retrieved 1/2 statements.
# Partially parsed test_text_constructor_custom_locale. Retrieved 2/3 statements.
# Partially parsed test_text_constructor_emojis_loaded. Retrieved 4/5 statements.
# Partially parsed test_text_constructor_dataset_loaded. Retrieved 4/5 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = var_0._emojis
    var_3 = len(var_2)

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._dataset
    var_2 = var_0._dataset
    var_3 = len(var_2)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_text_constructor. Retrieved 4/6 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_text_constructor_initializes_with_default_locale. Retrieved 1/2 statements.
# Partially parsed test_text_constructor_initializes_with_specified_locale. Retrieved 2/3 statements.
# Partially parsed test_text_constructor_loads_emojis. Retrieved 4/5 statements.
# Partially parsed test_text_constructor_inherits_from_base_data_provider. Retrieved 1/2 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = var_0._emojis
    var_3 = len(var_2)

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Text()



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = len(var_1)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_text_constructor. Retrieved 10/13 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis
    var_4 = var_0.Meta
    var_5 = 'name'
    var_6 = hasattr(var_4, var_5)
    var_7 = var_0.Meta
    var_8 = 'datafile'
    var_9 = hasattr(var_7, var_8)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_text_constructor. Retrieved 10/13 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis
    var_4 = 'locale'
    var_5 = hasattr(var_0, var_4)
    var_6 = var_0.locale
    var_7 = '_dataset'
    var_8 = hasattr(var_0, var_7)
    var_9 = var_0._dataset

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Text()
    var_2 = 'ru'
    var_3 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Text()
    var_2 = module_0.Text()
    var_3 = var_1.word()
    var_4 = var_2.word()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_text_constructor. Retrieved 4/6 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = 'Meta'
    var_3 = hasattr(var_0, var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test__read_global_file_returns_non_empty_dict. Retrieved 2/3 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_alphabet_returns_uppercase_by_default. Retrieved 2/5 statements.
# Partially parsed test_alphabet_returns_lowercase_when_specified. Retrieved 3/6 statements.
# Partially parsed test_alphabet_returns_only_letters. Retrieved 2/4 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0.alphabet()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = True
    var_2 = var_0.alphabet(var_1)

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0.alphabet()
    var_2 = len(var_1)

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0.alphabet()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_text_constructor. Retrieved 4/7 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_text_constructor. Retrieved 4/7 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis



# Parsed testcases at query #4
#--------------------------




import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Text()



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Text()
    var_2 = var_1.Meta
    var_3 = 'datafile'
    var_4 = hasattr(var_2, var_3)



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = bool(var_1)
    assert var_2 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_text_constructor_default_locale. Retrieved 1/2 statements.
# Partially parsed test_text_constructor_custom_locale. Retrieved 2/3 statements.
# Partially parsed test_text_constructor_emojis_loaded. Retrieved 4/5 statements.
# Partially parsed test_text_constructor_dataset_loaded. Retrieved 4/5 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Text()
    var_2 = module_0.Text()
    var_3 = var_1.word()
    var_4 = var_2.word()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis
    var_2 = var_0._emojis
    var_3 = len(var_2)

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._dataset
    var_2 = var_0._dataset
    var_3 = len(var_2)



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_emojis_initialized_without_error. Retrieved 2/3 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = var_0._emojis



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_text_constructor. Retrieved 4/8 statements.
# Partially parsed test_text_constructor_with_locale. Retrieved 2/3 statements.


import mimesis.providers.text as module_0

def test_case_0():
    var_0 = module_0.Text()
    var_1 = '_emojis'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._emojis

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Text()

import mimesis.providers.text as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Text()



