####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_9 = 'unsupported_format'
    var_10 = module_0.String(format=var_9)
    var_11 = 'number'
    var_12 = module_0.String(format=var_11)
    var_13 = module_0.Boolean()
    var_14 = 'a'
    var_15 = 'b'
    var_16 = [var_14, var_15]
    var_17 = module_0.Choice(choices=var_16)



# Parsed testcases at query #2
#--------------------------


def test_case_0():
    var_0 = 'Test Label'
    var_1 = 'test_field'
    var_2 = 'hello'
    var_3 = None
    var_4 = 'forms/textarea.html'
    var_5 = 0
    var_6 = 'password'
    var_7 = 'secret'
    var_8 = 'password123'
    var_9 = 'error_field'
    var_10 = 'This field is required'
    var_11 = 'choice_field'
    var_12 = 'option1'
    var_13 = 'forms/select.html'



# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'A'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'B'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = module_0.Choice(choices=var_6)
    var_8 = module_0.Boolean()
    var_9 = 'text'
    var_10 = module_0.String(format=var_9)
    var_11 = 'email'
    var_12 = module_0.String(format=var_11)
    var_13 = module_0.Field()
    var_14 = module_0.Object()



# Parsed testcases at query #4
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'John'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'name'
    var_2 = 'active'
    var_3 = 'role'
    var_4 = 'email'
    var_5 = 'John Doe'
    var_6 = True
    var_7 = 'admin'
    var_8 = 'john@example.com'
    var_9 = {var_1: var_5, var_2: var_6, var_3: var_7, var_4: var_8}
    var_10 = 'Invalid Name'
    var_11 = {var_1: var_10}
    var_12 = 'Field is required'
    var_13 = False
    var_14 = True
    assert var_14 is True
    var_15 = module_0.String()
    var_16 = 'visible'
    var_17 = 'hidden'
    var_18 = 'yes'
    var_19 = 'no'
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = {var_16: var_18, var_17: var_19}



# Parsed testcases at query #6
#--------------------------




# Parsed testcases at query #7
#--------------------------




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
    var_11 = module_0.Field()



# Parsed testcases at query #9
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = module_0.Boolean()
    var_4 = 'text'
    var_5 = module_0.String(format=var_4)
    var_6 = 'email'
    var_7 = module_0.String(format=var_6)
    var_8 = module_0.Field()
    var_9 = 'name'
    var_10 = module_0.String()
    var_11 = {var_9: var_10}



# Parsed testcases at query #10
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
    var_12 = module_0.Object()



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


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = '/tmp/templates'
    var_3 = module_0.Jinja2Forms(directory=var_2)
    var_4 = 'my_app'
    var_5 = module_0.Jinja2Forms(package=var_4)
    var_6 = 'templates'
    var_7 = module_0.Jinja2Forms(directory=var_2, package=var_4)
    var_8 = '/tmp'
    var_9 = module_0.Jinja2Forms(directory=var_8)
    var_10 = 'name'
    var_11 = 'test'
    var_12 = {var_10: var_11}
    var_13 = {var_10: var_11}



# Parsed testcases at query #13
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
    var_7 = 'unknown_format'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.Boolean()
    var_10 = 'a'
    var_11 = 'b'
    var_12 = [var_10, var_11]
    var_13 = module_0.Choice(choices=var_12)



# Parsed testcases at query #14
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
    var_8 = None
    var_9 = (var_6, var_8)
    var_10 = 'not-an-integer'
    var_11 = {var_2: var_4, var_3: var_10}
    var_12 = 'Must be an integer'
    var_13 = {var_3: var_12}
    var_14 = {}
    var_15 = (var_6, var_13)
    var_16 = {}



# Parsed testcases at query #15
#--------------------------




# Parsed testcases at query #16
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = 'text'
    var_2 = 'Bio'
    var_3 = module_0.String(format=var_1)
    var_4 = 'bio'
    var_5 = 'Hello World'
    var_6 = None
    var_7 = 'forms/textarea.html'
    var_8 = 'field_id'
    var_9 = 'field_name'
    var_10 = 'field'
    var_11 = 'label'
    var_12 = 'required'
    var_13 = 'input_type'
    var_14 = 'value'
    var_15 = 'error'
    var_16 = True
    var_17 = {var_8: var_4, var_9: var_4, var_10: var_3, var_11: var_2, var_12: var_16, var_13: var_1, var_14: var_5, var_15: var_6}
    var_18 = 'password'
    var_19 = module_0.String(format=var_18)
    var_20 = 'user_password'
    var_21 = 'secret123'
    var_22 = 'user-password'
    var_23 = ''
    var_24 = {var_8: var_22, var_9: var_20, var_10: var_19, var_11: var_20, var_12: var_16, var_13: var_18, var_14: var_23, var_15: var_6}
    var_25 = 'active'
    var_26 = 'is_active'
    var_27 = 'forms/checkbox.html'
    var_28 = 'cat'
    var_29 = 'category'
    var_30 = 'A'
    var_31 = 'Invalid choice'
    var_32 = 'forms/select.html'
    var_33 = 'email_addr'
    var_34 = 'email'
    var_35 = 'test@example.com'
    var_36 = 'email-addr'



####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
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
    var_7 = 'color'
    var_8 = {var_1: var_1, var_2: var_2, var_3: var_3, var_4: var_4, var_5: var_5, var_6: var_6, var_7: var_7}
    var_9 = 'unsupported_format'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Boolean()
    var_12 = '1'
    var_13 = 'One'
    var_14 = (var_12, var_13)
    var_15 = [var_14]
    var_16 = module_0.Choice(choices=var_15)



# Parsed testcases at query #2
#--------------------------




# Parsed testcases at query #3
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'John'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #4
#--------------------------


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_field'
    var_2 = 'hello'
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
    var_13 = 'test-field'
    var_14 = 'Test Label'
    var_15 = True
    var_16 = 'text'
    var_17 = 'hello'
    var_18 = None
    var_19 = 'password_field'
    var_20 = 'secret123'
    var_21 = 'forms/input.html'
    var_22 = 'choice_field'
    var_23 = 'option1'
    var_24 = 'error message'
    var_25 = 'forms/select.html'



# Parsed testcases at query #5
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = module_0.Choice(choices=var_3)
    var_5 = module_0.Boolean()
    var_6 = 'text'
    var_7 = module_0.String(format=var_6)
    var_8 = 'email'
    var_9 = module_0.String(format=var_8)
    var_10 = module_0.Integer()
    var_11 = module_0.String()



# Parsed testcases at query #6
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
    var_7 = module_0.String()
    var_8 = module_0.Field()
    var_9 = module_0.Object()



# Parsed testcases at query #7
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



# Parsed testcases at query #8
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



# Parsed testcases at query #9
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
    var_10 = 'Initial'
    var_11 = 20
    var_12 = {var_2: var_10, var_3: var_11}
    var_13 = None



# Parsed testcases at query #10
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



# Parsed testcases at query #11
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'date'
    var_4 = module_0.String(format=var_3)
    var_5 = 'password'
    var_6 = module_0.String(format=var_5)
    var_7 = 'unknown_format'
    var_8 = module_0.String(format=var_7)
    var_9 = module_0.Boolean()
    var_10 = 1
    var_11 = 'One'
    var_12 = (var_10, var_11)
    var_13 = 2
    var_14 = 'Two'
    var_15 = (var_13, var_14)
    var_16 = [var_12, var_15]
    var_17 = module_0.Choice(choices=var_16)



# Parsed testcases at query #12
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = module_0.Object()
    var_3 = 'category'
    var_4 = 'active'
    var_5 = 'bio'
    var_6 = 'name'
    var_7 = 'metadata'



# Parsed testcases at query #13
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Boolean()
    var_1 = module_0.Integer()
    var_2 = 'name'
    var_3 = 'name'
    var_4 = 'John Doe'
    var_5 = None
    var_6 = 'forms/textarea.html'
    var_7 = 'field_id'
    var_8 = 'field_name'
    var_9 = 'field'
    var_10 = 'label'
    var_11 = 'required'
    var_12 = 'input_type'
    var_13 = 'value'
    var_14 = 'error'
    var_15 = True
    var_16 = 'text'
    var_17 = 'is_active'
    var_18 = 'is_active'
    var_19 = 'Error!'
    var_20 = 'forms/checkbox.html'
    var_21 = 'is-active'
    var_22 = 'category'
    var_23 = 'category'
    var_24 = 'A'
    var_25 = 'forms/select.html'
    var_26 = 'pwd'
    var_27 = 'secret123'



# Parsed testcases at query #14
#--------------------------


def test_case_0():
    var_0 = 'username'
    var_1 = 'johndoe'
    var_2 = None
    var_3 = 'forms/textarea.html'
    var_4 = 'field_id'
    var_5 = 'field_name'
    var_6 = 'field'
    var_7 = 'label'
    var_8 = 'required'
    var_9 = 'input_type'
    var_10 = 'value'
    var_11 = 'error'
    var_12 = 'username'
    var_13 = 'Test Label'
    var_14 = True
    var_15 = 'text'
    var_16 = 'johndoe'
    var_17 = None
    var_18 = 'user_password'
    var_19 = 'secret123'
    var_20 = 0
    var_21 = 'This field is required'
    var_22 = 'email'



# Parsed testcases at query #15
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.String()
    var_1 = module_0.Boolean()
    var_2 = 'name'
    var_3 = 'active'
    var_4 = 'John'
    var_5 = True
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 0



# Parsed testcases at query #16
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = 'templates'
    var_1 = module_0.Jinja2Forms(directory=var_0)
    var_2 = module_0.Jinja2Forms()
    var_3 = '/tmp/templates'
    var_4 = module_0.Jinja2Forms(directory=var_3)
    var_5 = 'my_app'
    var_6 = module_0.Jinja2Forms(package=var_5)
    var_7 = 'templates'
    var_8 = '/tmp/templates'
    var_9 = 'my_app'
    var_10 = module_0.Jinja2Forms(directory=var_8, package=var_9)
    var_11 = 'templates'



# Parsed testcases at query #17
#--------------------------


import typesystem.fields as module_0

def test_case_0():
    var_0 = module_0.Field()
    var_1 = 'email'
    var_2 = module_0.String(format=var_1)
    var_3 = 'custom_format'
    var_4 = module_0.String(format=var_3)
    var_5 = 'password'
    var_6 = module_0.String(format=var_5)
    var_7 = 'date'
    var_8 = module_0.String(format=var_7)
    var_9 = 'number'
    var_10 = module_0.String(format=var_9)
    var_11 = module_0.Boolean()



# Parsed testcases at query #18
#--------------------------


import typesystem.forms as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Jinja2Forms(directory=var_0, package=var_0)
    var_2 = '/tmp/templates'
    var_3 = None
    var_4 = module_0.Jinja2Forms(directory=var_2, package=var_3)
    var_5 = 'my_app'
    var_6 = module_0.Jinja2Forms(directory=var_3, package=var_5)
    var_7 = 'templates'
    var_8 = module_0.Jinja2Forms(directory=var_2, package=var_5)
    var_9 = 0



