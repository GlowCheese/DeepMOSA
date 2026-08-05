####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = 'forms/textarea.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = True
    var_14 = 'text'
    var_15 = 'password'
    var_16 = module_0.String(format=var_15)
    var_17 = 'secret123'
    var_18 = 0
    var_19 = 'active'
    var_20 = 'forms/checkbox.html'
    var_21 = 'role'
    var_22 = 'admin'
    var_23 = 'Invalid choice'
    var_24 = 'forms/select.html'
    var_25 = module_0.String()
    var_26 = 'user_email'
    var_27 = 'test@test.com'



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Choice(choices=var_4)
    var_6 = module_0.Boolean()
    var_7 = 'text'
    var_8 = module_0.String(format=var_7)
    var_9 = 'email'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Field()
    var_12 = 'name'
    var_13 = module_0.String()
    var_14 = {var_12: var_13}



# Parsed testcases at query #3
#--------------------------


def test_case_0():
    var_0 = 'Field One'
    var_1 = 'Read Only Field'
    var_2 = True
    var_3 = 'Field Three'
    var_4 = 'f1'
    var_5 = 'f2'
    var_6 = 'f3'
    var_7 = 'val1'
    var_8 = 'val3'
    var_9 = {var_4: var_7, var_6: var_8}
    var_10 = {var_4: var_7, var_6: var_8}
    var_11 = 'invalid'
    var_12 = {var_4: var_11, var_6: var_8}
    var_13 = 'old'
    var_14 = {var_4: var_13}
    var_15 = 'new'
    var_16 = 'new_val'
    var_17 = {var_4: var_15, var_6: var_16}
    var_18 = False
    var_19 = True
    assert var_19 is True



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'password'
    var_4 = module_0.String(format=var_3)
    var_5 = 'date'
    var_6 = module_0.String(format=var_5)
    var_7 = 'unsupported_format'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.Boolean()
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)



# Parsed testcases at query #5
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = '/tmp/templates'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'my_app'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = 'templates'
    var_6 = module_0.Jinja2Forms(directory=var_1, package=var_3)
    var_7 = 0



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'password'
    var_4 = module_0.String(format=var_3)
    var_5 = 'date'
    var_6 = module_0.String(format=var_5)
    var_7 = 'unsupported-format'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.Boolean()
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'A'
    var_3 = 'B'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Choice(choices=var_4)
    var_6 = module_0.Boolean()
    var_7 = 'text'
    var_8 = module_0.String(format=var_7)
    var_9 = 'email'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Integer()
    var_12 = 'name'
    var_13 = module_0.String()
    var_14 = {var_12: var_13}



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'Test'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'name'
    var_2 = 'test_user'
    var_3 = 'This field is required'
    var_4 = 'forms/textarea.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = 'Username'
    var_14 = True
    var_15 = 'text'
    var_16 = 'password'
    var_17 = module_0.String(format=var_16)
    var_18 = 'PwSchema'
    var_19 = 'pw'
    var_20 = {var_19: var_17}
    var_21 = 'secret123'
    var_22 = None
    var_23 = 0



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = 'forms/textarea.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = 'name'
    var_14 = 'User Name'
    var_15 = True
    var_16 = 'text'
    var_17 = None
    var_18 = 'active'
    var_19 = True
    var_20 = 'forms/checkbox.html'
    var_21 = 'active'
    var_22 = 'role'
    var_23 = 'admin'
    var_24 = 'Invalid choice'
    var_25 = 'forms/select.html'
    var_26 = 'role'
    var_27 = 'password'
    var_28 = 'secret123'
    var_29 = 'forms/input.html'
    var_30 = 0
    var_31 = module_0.String()
    var_32 = 'user_id'
    var_33 = '123'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'readonly_field'
    var_2 = 'Initial Value'
    var_3 = {var_0: var_2}
    var_4 = 'New Value'
    var_5 = 'Error Message'
    var_6 = {var_0: var_4}
    var_7 = {var_0: var_5}
    var_8 = {var_0: var_4}

def test_case_0():
    var_0 = 'name'
    var_1 = {}



# Parsed testcases at query #14
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = '/tmp/templates'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = 'loader'
    var_5 = 'myapp'
    var_6 = module_0.Jinja2Forms(package=var_5)
    var_7 = module_0.Jinja2Forms(directory=var_2, package=var_5)

import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = '/test'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = 'mypkg'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = 'templates'
    var_7 = module_0.Jinja2Forms(directory=var_2, package=var_4)



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'is_active'
    var_3 = 'category'
    var_4 = 'John Doe'
    var_5 = True
    var_6 = 'A'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_9 = ''
    var_10 = 'not-a-bool'
    var_11 = 'C'
    var_12 = {var_1: var_9, var_2: var_10, var_3: var_11}
    var_13 = 'This field is required'
    var_14 = 'Invalid boolean'
    var_15 = False
    var_16 = True
    var_17 = module_0.String()
    var_18 = 'visible_info'
    var_19 = 'hi'
    var_20 = {var_18: var_19}



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'test'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'email'
    var_2 = 'password'
    var_3 = 'number'
    var_4 = 'date'
    var_5 = 'url'
    var_6 = 'tel'
    var_7 = 'color'
    var_8 = {var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6, var_7: var_7}
    var_9 = 'unsupported_format'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Boolean()
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]
    var_15 = module_0.Choice(choices=var_14)



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'text'
    var_2 = module_0.String(format=var_1)
    var_3 = 'email'
    var_4 = module_0.String(format=var_3)
    var_5 = 'password'
    var_6 = module_0.String(format=var_5)
    var_7 = 'date'
    var_8 = module_0.String(format=var_7)
    var_9 = 'unsupported-format'
    var_10 = module_0.String(format=var_9)
    var_11 = 'number'
    var_12 = module_0.String(format=var_11)
    var_13 = module_0.Boolean()



# Parsed testcases at query #19
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = 'forms/input.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = 'name'
    var_14 = 'Full Name'
    var_15 = True
    var_16 = 'text'
    var_17 = 'is_active'
    var_18 = 'Error!'
    var_19 = 'forms/checkbox.html'
    var_20 = 'is-active'
    var_21 = 'is_active'
    var_22 = 'category'
    var_23 = 'a'
    var_24 = 'forms/select.html'
    var_25 = 'bio'
    var_26 = 'Hello'
    var_27 = 'forms/textarea.html'
    var_28 = 'password'
    var_29 = 'secret123'
    var_30 = module_0.String()
    var_31 = 'user_profile_id'
    var_32 = '123'



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'Alice'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = {}
    var_11 = {}



# Parsed testcases at query #21
#--------------------------




# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'test_field'
    var_3 = 'is_active'
    var_4 = 'hello'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}

import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'test_field'
    var_2 = ''
    var_3 = {var_1: var_2}
    var_4 = {var_1: var_2}
    var_5 = 'This field is required'



# Parsed testcases at query #23
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'date'
    var_4 = module_0.String(format=var_3)
    var_5 = 'unsupported-format'
    var_6 = module_0.String(format=var_5)
    var_7 = module_0.Boolean()
    var_8 = module_0.Integer()
    var_9 = 'password'
    var_10 = module_0.String(format=var_9)
    var_11 = 'number'
    var_12 = module_0.String(format=var_11)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'not-an-integer'
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = None



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = 'user_name'
    var_3 = 'jdoe'
    var_4 = None
    var_5 = 'forms/textarea.html'
    var_6 = 'field_id'
    var_7 = 'field_name'
    var_8 = 'field'
    var_9 = 'label'
    var_10 = 'required'
    var_11 = 'input_type'
    var_12 = 'value'
    var_13 = 'error'
    var_14 = 'user-name'
    var_15 = 'Username'
    var_16 = True
    var_17 = {var_6: var_14, var_7: var_2, var_8: var_1, var_9: var_15, var_10: var_16, var_11: var_0, var_12: var_3, var_13: var_4}
    var_18 = 'password'
    var_19 = module_0.String(format=var_18)
    var_20 = 'secret123'
    var_21 = ''
    var_22 = {var_12: var_21}
    var_23 = module_0.Boolean()
    var_24 = 'terms'
    var_25 = 'Required'
    var_26 = 'forms/checkbox.html'
    var_27 = {var_6: var_24, var_13: var_25}
    var_28 = 'A'
    var_29 = 'B'
    var_30 = [var_28, var_29]
    var_31 = 'my_choice'
    var_32 = 'forms/select.html'
    var_33 = 'email'
    var_34 = module_0.String(format=var_33)
    var_35 = 'email_address'
    var_36 = 'test@example.com'
    var_37 = 'forms/input.html'
    var_38 = 'Email Address'
    var_39 = {var_11: var_33, var_9: var_38}



# Parsed testcases at query #26
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'some/path'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = '/tmp/templates'
    var_4 = module_0.Jinja2Forms(directory=var_3)
    var_5 = 'loader'
    var_6 = 'my_app'
    var_7 = module_0.Jinja2Forms(package=var_6)
    var_8 = 'templates'
    var_9 = 'loader'
    var_10 = '/tmp/templates'
    var_11 = 'my_app'
    var_12 = module_0.Jinja2Forms(directory=var_10, package=var_11)
    var_13 = 0



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'is_active'
    var_4 = 'Test'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #2
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = {}
    var_11 = {}
    var_12 = None



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'test'
    var_5 = True



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'A'
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 'B'
    var_6 = (var_4, var_5)
    var_7 = [var_3, var_6]
    var_8 = module_0.Choice(choices=var_7)
    var_9 = module_0.Boolean()
    var_10 = 'text'
    var_11 = module_0.String(format=var_10)
    var_12 = 'email'
    var_13 = module_0.String(format=var_12)
    var_14 = module_0.Field()
    var_15 = module_0.String()



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John Doe'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = {}
    var_10 = None
    var_11 = (var_8, var_10)
    var_12 = ''
    var_13 = 'not-an-int'
    var_14 = {var_0: var_12, var_1: var_13}
    var_15 = 'cannot be empty'
    var_16 = 'must be an integer'
    var_17 = {var_0: var_15, var_1: var_16}
    var_18 = {}
    var_19 = {var_0: var_12}
    var_20 = (var_19, var_17)
    var_21 = {}



# Parsed testcases at query #6
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John Doe'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = None
    var_10 = 'not-an-integer'
    var_11 = {var_0: var_6, var_1: var_10}



# Parsed testcases at query #7
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()

import typesystem.fields as module_0

def test_case_0():
    var_0 = '\n    Integration-style test to ensure __html__ correctly \n    wraps the actual rendered output of fields.\n    '
    var_1 = "<input type='text' name='name'>"
    var_2 = "<input type='checkbox' name='active'>"
    var_3 = module_0.String()
    var_4 = module_0.Boolean()
    var_5 = "<input type='text' name='name'><input type='checkbox' name='active'>"



# Parsed testcases at query #8
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'email'
    var_3 = 'active'
    var_4 = 'category'
    var_5 = 'John Doe'
    var_6 = 'john@example.com'
    var_7 = True
    var_8 = 'A'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_11 = 'invalid-email'
    var_12 = {var_1: var_5, var_2: var_11}
    var_13 = 'Invalid email format'
    var_14 = {var_2: var_13}
    var_15 = False
    var_16 = True



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'age'
    var_3 = 'is_active'
    var_4 = 'role'
    var_5 = 'John'
    var_6 = '30'
    var_7 = True
    var_8 = 'admin'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'This field is required'
    var_11 = 'Must be a number'
    var_12 = {var_1: var_10, var_2: var_11}
    var_13 = ''
    var_14 = 'abc'
    var_15 = {var_1: var_13, var_2: var_14, var_3: var_7, var_4: var_8}
    var_16 = False
    var_17 = True
    var_18 = module_0.String()
    var_19 = 'visible'
    var_20 = 'hidden_field'
    var_21 = 'hi'
    var_22 = 'secret'
    var_23 = {var_19: var_21, var_20: var_22}



# Parsed testcases at query #10
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'date'
    var_4 = module_0.String(format=var_3)
    var_5 = 'unsupported-format'
    var_6 = module_0.String(format=var_5)
    var_7 = 'password'
    var_8 = module_0.String(format=var_7)
    var_9 = 'number'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Boolean()
    var_12 = 'a'
    var_13 = 'b'
    var_14 = [var_12, var_13]



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = 'forms/textarea.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = True
    var_14 = 'text'
    var_15 = 'pwd'
    var_16 = 'secret123'
    var_17 = 'role'
    var_18 = 'admin'
    var_19 = 'Invalid choice'
    var_20 = 'forms/select.html'
    var_21 = module_0.String()
    var_22 = 'first_name'
    var_23 = 'Test'



# Parsed testcases at query #13
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'is_active'
    var_5 = 'bio'
    var_6 = 'text'
    var_7 = 'email'
    var_8 = 'generic'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'test_field_name'
    var_1 = 'test-field-name'
    var_2 = 'hello world'
    var_3 = 'This is an error'
    var_4 = 'forms/textarea.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = 'Test Label'
    var_14 = False
    var_15 = 'text'

def test_case_0():
    var_0 = 'password'
    var_1 = 'secret123'
    var_2 = None



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'email'
    var_2 = 'password'
    var_3 = 'number'
    var_4 = 'date'
    var_5 = 'url'
    var_6 = 'tel'
    var_7 = {var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6}
    var_8 = 'unsupported-type'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.Boolean()
    var_11 = 'a'
    var_12 = 'A'
    var_13 = (var_11, var_12)
    var_14 = [var_13]
    var_15 = module_0.Choice(choices=var_14)



# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.Boolean()
    var_5 = 'text'
    var_6 = module_0.String(format=var_5)
    var_7 = module_0.String()
    var_8 = 'email'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.String()



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'John Doe'
    var_4 = None
    var_5 = 'forms/input.html'
    var_6 = 'field_id'
    var_7 = 'field_name'
    var_8 = 'field'
    var_9 = 'label'
    var_10 = 'required'
    var_11 = 'input_type'
    var_12 = 'value'
    var_13 = 'error'
    var_14 = 'name'
    var_15 = True
    var_16 = 'text'
    var_17 = 'is_active'
    var_18 = 'forms/checkbox.html'
    var_19 = 'is-active'
    var_20 = 'is_active'
    var_21 = 'category'
    var_22 = 'a'
    var_23 = 'Invalid choice'
    var_24 = 'forms/select.html'
    var_25 = 'category'
    var_26 = 'password'
    var_27 = 'secret123'
    var_28 = 'password'
    var_29 = ''
    var_30 = 'email'
    var_31 = 'test@example.com'
    var_32 = 'email'



# Parsed testcases at query #18
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'text'
    var_1 = module_0.String(format=var_0)
    var_2 = 'description'
    var_3 = 'hello'
    var_4 = None
    var_5 = 'forms/textarea.html'
    var_6 = 'field_id'
    var_7 = 'field_name'
    var_8 = 'field'
    var_9 = 'label'
    var_10 = 'required'
    var_11 = 'input_type'
    var_12 = 'value'
    var_13 = 'error'
    var_14 = True
    var_15 = {var_6: var_2, var_7: var_2, var_8: var_1, var_9: var_2, var_10: var_14, var_11: var_0, var_12: var_3, var_13: var_4}
    var_16 = module_0.Boolean()
    var_17 = 'is_active'
    var_18 = 'Error!'
    var_19 = 'forms/checkbox.html'
    var_20 = 'is-active'
    var_21 = {var_6: var_20, var_7: var_17, var_8: var_16, var_9: var_17, var_10: var_14, var_11: var_0, var_12: var_14, var_13: var_18}
    var_22 = 'a'
    var_23 = 'b'
    var_24 = [var_22, var_23]
    var_25 = module_0.Choice(choices=var_24)
    var_26 = 'password'
    var_27 = module_0.String(format=var_26)
    var_28 = 'secret'
    var_29 = '12345'
    var_30 = 'forms/select.html'
    var_31 = 0
    var_32 = 'Custom Label'
    var_33 = module_0.String()
    var_34 = 'user_name'
    var_35 = 'user-name'
    var_36 = False
    var_37 = {var_6: var_35, var_7: var_34, var_8: var_33, var_9: var_32, var_10: var_36, var_11: var_0, var_12: var_4, var_13: var_4}



# Parsed testcases at query #19
#--------------------------


def test_case_0():
    var_0 = 'name'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'is_active'
    var_5 = 'bio'
    var_6 = 'email'
    var_7 = 'user'



# Parsed testcases at query #20
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = {}
    var_11 = {var_2: var_4}
    var_12 = {}
    var_13 = {}



# Parsed testcases at query #21
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {}
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = {}
    var_11 = {}



# Parsed testcases at query #22
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()



# Parsed testcases at query #23
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = module_0.Jinja2Forms()
    var_1 = '/tmp/templates'
    var_2 = module_0.Jinja2Forms(directory=var_1)
    var_3 = 'my_app'
    var_4 = module_0.Jinja2Forms(package=var_3)
    var_5 = 'templates'
    var_6 = module_0.Jinja2Forms(directory=var_1, package=var_3)
    var_7 = 0
    var_8 = 'd'
    var_9 = 'p'
    var_10 = module_0.Jinja2Forms(directory=var_8, package=var_9)
    var_11 = '/tmp'
    var_12 = module_0.Jinja2Forms(directory=var_11)



# Parsed testcases at query #24
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'John Doe'
    var_3 = None
    var_4 = 'forms/input.html'
    var_5 = 'field_id'
    var_6 = 'field_name'
    var_7 = 'field'
    var_8 = 'label'
    var_9 = 'required'
    var_10 = 'input_type'
    var_11 = 'value'
    var_12 = 'error'
    var_13 = 'name'
    var_14 = 'Full Name'
    var_15 = True
    var_16 = 'text'
    var_17 = 'is_active'
    var_18 = 'forms/checkbox.html'
    var_19 = 'category'



# Parsed testcases at query #25
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John Doe'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = None
    var_8 = 'not-an-integer'
    var_9 = {var_2: var_4, var_3: var_8}



# Parsed testcases at query #26
#--------------------------


import typesystem.fields as module_0
import typesystem.schemas as module_1

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.String()
    var_3 = module_0.Integer()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = None
    var_10 = (var_8, var_9)
    var_11 = ''
    var_12 = 'not-an-int'
    var_13 = {var_0: var_11, var_1: var_12}
    var_14 = 'Cannot be empty'
    var_15 = 'Must be an integer'
    var_16 = {var_0: var_14, var_1: var_15}
    var_17 = (var_8, var_16)



# Parsed testcases at query #27
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()



