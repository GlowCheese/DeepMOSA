####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = var_0._pathlib_home
    var_3 = str(var_2)
    var_4 = 'linux'
    var_5 = module_0.Path(var_4)
    var_6 = var_5._pathlib_home
    var_7 = var_5._pathlib_home
    var_8 = str(var_7)
    var_9 = 'win32'
    var_10 = module_0.Path(var_9)
    var_11 = var_10._pathlib_home
    var_12 = var_10._pathlib_home
    var_13 = str(var_12)
    var_14 = 'freebsd12'
    var_15 = module_0.Path(var_14)
    var_16 = var_15._pathlib_home
    var_17 = var_15._pathlib_home
    var_18 = str(var_17)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'
    var_4 = '/'
    var_5 = 'win32'
    var_6 = module_0.Path(var_5)
    var_7 = var_6.user()
    var_8 = 'C:\\Users\\'
    var_9 = '\\'



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = 'freebsd12'
    var_3 = module_0.Path(var_2)
    var_4 = 'linux'
    var_5 = module_0.Path(var_4)
    var_6 = var_5._pathlib_home
    var_7 = str(var_6)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'
    var_4 = '/'
    var_5 = 'win32'
    var_6 = module_0.Path(var_5)
    var_7 = var_6.user()
    var_8 = '\\'
    var_9 = 'darwin'
    var_10 = module_0.Path(var_9)
    var_11 = var_10.user()
    var_12 = '/Users/'



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'
    var_4 = 6
    var_5 = var_2[var_4:]
    var_6 = 'win32'
    var_7 = module_0.Path(var_6)
    var_8 = var_7.user()
    var_9 = '\\Users\\'
    var_10 = 7
    var_11 = var_8[var_10:]
    var_12 = 'darwin'
    var_13 = module_0.Path(var_12)
    var_14 = var_13.user()
    var_15 = '/Users/'
    var_16 = var_14[var_10:]
    var_17 = 'freebsd'
    var_18 = module_0.Path(var_17)
    var_19 = var_18.user()
    var_20 = var_19[var_4:]



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'
    var_4 = -1
    var_5 = '/'
    var_6 = user_linux.split(var_5)[var_4]
    var_7 = 'win32'
    var_8 = module_0.Path(var_7)
    var_9 = var_8.user()
    var_10 = '\\Users\\'
    var_11 = -1
    var_12 = '\\'
    var_13 = user_windows.split(var_12)[var_11]
    var_14 = 'darwin'
    var_15 = module_0.Path(var_14)
    var_16 = var_15.user()
    var_17 = '/Users/'
    var_18 = -1
    var_19 = user_darwin.split(var_5)[var_18]



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = var_0._pathlib_home
    var_3 = var_0._pathlib_home
    var_4 = 'freebsd12'
    var_5 = module_0.Path(var_4)
    var_6 = var_5._pathlib_home
    var_7 = 'linux'
    var_8 = module_0.Path(var_7)
    var_9 = var_8._pathlib_home
    var_10 = str(var_9)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = 'linux'
    var_3 = module_0.Path(var_2)
    var_4 = var_3._pathlib_home
    var_5 = var_3._pathlib_home
    var_6 = str(var_5)
    var_7 = 'darwin'
    var_8 = module_0.Path(var_7)
    var_9 = var_8._pathlib_home
    var_10 = var_8._pathlib_home
    var_11 = str(var_10)
    var_12 = 'win32'
    var_13 = module_0.Path(var_12)
    var_14 = var_13._pathlib_home
    var_15 = var_13._pathlib_home
    var_16 = str(var_15)
    var_17 = 'win64'
    var_18 = module_0.Path(var_17)
    var_19 = var_18._pathlib_home
    var_20 = var_18._pathlib_home
    var_21 = str(var_20)
    var_22 = 'freebsd'
    var_23 = module_0.Path(var_22)
    var_24 = var_23._pathlib_home
    var_25 = var_23._pathlib_home
    var_26 = str(var_25)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home'
    var_4 = '/'
    var_5 = 'win32'
    var_6 = module_0.Path(var_5)
    var_7 = var_6.user()
    var_8 = 'C:\\Users'
    var_9 = '\\'
    var_10 = 'darwin'
    var_11 = module_0.Path(var_10)
    var_12 = var_11.user()
    var_13 = '/Users'
    var_14 = 'freebsd'
    var_15 = module_0.Path(var_14)
    var_16 = var_15.user()



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home/'
    var_4 = '/'
    var_5 = 'win32'
    var_6 = module_0.Path(var_5)
    var_7 = var_6.user()
    var_8 = -1
    var_9 = '\\'
    var_10 = user_win.split(var_9)[var_8]
    var_11 = 'darwin'
    var_12 = module_0.Path(var_11)
    var_13 = var_12.user()
    var_14 = '/Users/'
    var_15 = 'freebsd'
    var_16 = module_0.Path(var_15)
    var_17 = var_16.user()



