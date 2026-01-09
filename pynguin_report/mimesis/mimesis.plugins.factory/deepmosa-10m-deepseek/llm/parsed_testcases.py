####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.


import mimesis.plugins.factory as module_0


def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'data1'
    var_2 = 'data2'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'test_field'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #7
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #12
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #13
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #14
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #17
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 2
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 2})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'test_field'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'



# Parsed testcases at query #20
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_locale_is_set_when_provided. Retrieved 1/4 statements.
# Partially parsed test_locale_and_kwargs_are_set_together. Retrieved 2/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.kwargs
    var_4 = bool(var_2.kwargs == {})
    assert var_4 is True


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'param1'
    var_4 = 'param2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.kwargs
    var_8 = bool(var_6.kwargs == {'param1': 'value1', 'param2': 'value2'})
    assert var_8 is True


def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'extra'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #23
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = 'Step'
    var_4 = ()
    var_5 = 'builder'
    var_6 = 'Builder'
    var_7 = ()
    var_8 = 'factory_meta'
    var_9 = 'Meta'
    var_10 = ()
    var_11 = 'declarations'
    var_12 = 'field_handlers'
    var_13 = []
    var_14 = {var_12: var_13}
    var_15 = {var_11: var_14}
    var_16 = type(var_9, var_10, var_15)
    var_17 = {var_8: var_16}
    var_18 = type(var_6, var_7, var_17)
    var_19 = {var_5: var_18}
    var_20 = type(var_3, var_4, var_19)
    var_21 = var_20()
    var_22 = None
    var_23 = None
    var_24 = var_2.evaluate(var_23, var_21, var_22)
    var_25 = bool(var_24 is not None)
    assert var_25 is True



# Parsed testcases at query #24
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #25
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_init_assigns_instance_variables. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'option'
    var_2 = True
    var_3 = {var_1: var_2}



# Parsed testcases at query #30
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = None
    var_5 = var_3 is var_4
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #32
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = None
    var_5 = var_3 is var_4
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'custom_arg'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'custom_arg': 'value'})
    assert var_8 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #2
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_locale_assigned_to_instance_variable. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'param'
    var_2 = 'data'
    var_3 = {var_1: var_2}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_init_sets_field_locale_and_kwargs. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'person.name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address.city'


def test_case_0():
    var_0 = 'text.word'
    var_1 = 5
    var_2 = 'length'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'text.word'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'length': 5})
    assert var_8 is True

def test_case_0():
    var_0 = 'person.full_name'
    var_1 = ' '



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #11
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'test_field'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'param'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = 'test_field'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_constructor_with_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value'
    var_2 = 'custom_arg'
    var_3 = {var_2: var_1}
    var_4 = module_0.FactoryField(var_0, **var_3)
    var_5 = var_4.field
    assert var_5 == 'test_field'
    var_6 = var_4.locale
    assert var_6 is None
    var_7 = var_4.kwargs
    var_8 = bool(var_4.kwargs == {'custom_arg': 'value'})
    assert var_8 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.full_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address.city'


def test_case_0():
    var_0 = 'text.word'
    var_1 = 5
    var_2 = True
    var_3 = 'length'
    var_4 = 'unique'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text.word'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'length': 5, 'unique': True})
    assert var_10 is True

def test_case_0():
    var_0 = 'person.email'
    var_1 = 'example.com'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_locale_is_assigned_to_instance_variable. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 1
    var_2 = 'two'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_locale_is_none_and_field_handlers_is_none. Retrieved 2/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_locale_is_set_when_provided. Retrieved 1/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = None
    var_2 = {}
    var_3 = module_0.FactoryField(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 is None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_init_assigns_field_locale_and_kwargs. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_locale_is_none_and_field_handlers_is_none. Retrieved 2/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 2/4 statements.



def test_case_0():
    var_0 = 'person.full_name'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'person.full_name'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'address.city'


def test_case_0():
    var_0 = 'text.word'
    var_1 = 5
    var_2 = True
    var_3 = 'length'
    var_4 = 'unique'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'text.word'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'length': 5, 'unique': True})
    assert var_10 is True

def test_case_0():
    var_0 = 'person.email'
    var_1 = 'example.com'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_locale_is_set_when_provided. Retrieved 1/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None

def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 4/6 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    var_4 = bool(var_2.field == var_0)
    assert var_4 is True
    var_5 = var_2.locale
    assert var_5 is None
    var_6 = var_2.kwargs
    var_7 = bool(var_2.kwargs == {})
    assert var_7 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 123
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.FactoryField(var_0, **var_8)
    var_10 = var_9.field
    var_11 = bool(var_9.field == var_0)
    assert var_11 is True
    var_12 = var_9.locale
    assert var_12 is None
    var_13 = var_9.kwargs
    var_14 = bool(var_9.kwargs == var_5)
    assert var_14 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'param'
    var_2 = 'data'
    var_3 = {var_1: var_2}



# Parsed testcases at query #28
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_init_assigns_field_locale_and_kwargs. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 6/8 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'
    var_6 = 'key1'
    var_7 = 'key2'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.FactoryField(var_5, **var_8)
    var_10 = var_9.field
    assert var_10 == 'test_field'
    var_11 = var_9.locale
    assert var_11 is None
    var_12 = var_9.kwargs
    var_13 = bool(var_9.kwargs == var_4)
    assert var_13 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_field'



# Parsed testcases at query #32
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #34
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = None
    var_5 = var_3 is var_4
    assert var_5 is True



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_locale_is_set_to_provided_value. Retrieved 1/4 statements.
# Partially parsed test_locale_is_set_to_provided_locale_object. Retrieved 1/4 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None

def test_case_0():
    var_0 = 'test_field'

def test_case_0():
    var_0 = 'test_field'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_factoryfield_constructor_with_custom_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'val1'
    var_2 = 'val2'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_locale_is_assigned_correctly_when_provided. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = {var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.locale
    assert var_7 is None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_locale_is_not_none_when_provided. Retrieved 2/6 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = None
    var_5 = var_3 is var_4
    assert var_5 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = None



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_factoryfield_constructor_with_locale. Retrieved 1/3 statements.
# Partially parsed test_factoryfield_constructor_with_locale_and_kwargs. Retrieved 3/5 statements.



def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.field
    assert var_3 == 'test_field'
    var_4 = var_2.locale
    assert var_4 is None
    var_5 = var_2.kwargs
    var_6 = bool(var_2.kwargs == {})
    assert var_6 is True

def test_case_0():
    var_0 = 'test_field'


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'
    var_3 = 'key1'
    var_4 = 'key2'
    var_5 = {var_3: var_1, var_4: var_2}
    var_6 = module_0.FactoryField(var_0, **var_5)
    var_7 = var_6.field
    assert var_7 == 'test_field'
    var_8 = var_6.locale
    assert var_8 is None
    var_9 = var_6.kwargs
    var_10 = bool(var_6.kwargs == {'key1': 'value1', 'key2': 'value2'})
    assert var_10 is True

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'value1'
    var_2 = 'value2'



# Parsed testcases at query #40
#--------------------------





def test_case_0():
    var_0 = 'test_field'
    var_1 = {}
    var_2 = module_0.FactoryField(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 is None



