####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'Test the constructor of class Path.'
    var_1 = 'linux'
    var_2 = module_0.Path(var_1)
    var_3 = var_2._pathlib_home
    var_4 = var_2._pathlib_home
    var_5 = str(var_4)
    var_6 = 'win32'
    var_7 = module_0.Path(var_6)
    var_8 = var_7._pathlib_home
    var_9 = var_7._pathlib_home
    var_10 = str(var_9)
    var_11 = 'freebsd'
    var_12 = module_0.Path(var_11)
    var_13 = var_12._pathlib_home
    var_14 = var_12._pathlib_home
    var_15 = str(var_14)



# Parsed testcases at query #2
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



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'linux'
    var_2 = module_0.Path(var_1)
    var_3 = 'win32'
    var_4 = module_0.Path(var_3)
    var_5 = 'darwin'
    var_6 = module_0.Path(var_5)
    var_7 = 'freebsd'
    var_8 = module_0.Path(var_7)
    var_9 = 'invalid'
    var_10 = module_0.Path(var_9)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'Test the user method of the Path class.'
    var_1 = 'win32'
    var_2 = module_0.Path(var_1)
    var_3 = var_2.user()
    var_4 = '\\'
    var_5 = 'linux'
    var_6 = module_0.Path(var_5)
    var_7 = var_6.user()
    var_8 = '/'
    var_9 = 'darwin'
    var_10 = module_0.Path(var_9)
    var_11 = var_10.user()
    var_12 = 'freebsd'
    var_13 = module_0.Path(var_12)
    var_14 = var_13.user()



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = 'linux'
    var_1 = 'darwin'
    var_2 = 'win32'
    var_3 = 'win64'
    var_4 = 'freebsd'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = any(var_2)
    var_7 = 0
    var_8 = 0
    var_9 = 'Test `test_Path_user` passed successfully.'
    var_10 = print(var_9)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = 'linux'
    var_3 = module_0.Path(var_2)
    var_4 = 'darwin'
    var_5 = module_0.Path(var_4)
    var_6 = 'freebsd'
    var_7 = module_0.Path(var_6)
    var_8 = var_1.user()
    var_9 = '\\Users\\'
    var_10 = var_3.user()
    var_11 = '/home/'
    var_12 = var_5.user()
    var_13 = '/Users/'
    var_14 = var_7.user()
    var_15 = '/usr/home/'



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'freebsd10'
    var_2 = module_0.Path(var_1)
    var_3 = 'invalid_platform'
    var_4 = module_0.Path(var_3)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.path as module_0
import pathlib as module_1

def test_case_0():
    var_0 = module_0.Path()
    var_1 = module_1.PurePosixPath()
    var_2 = 'home'



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.path as module_0

def test_case_0():
    var_0 = 'linux'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '/home'
    var_4 = -1
    var_5 = '/'
    var_6 = user_path.split(var_5)[var_4]
    var_7 = 'win32'
    var_8 = module_0.Path(var_7)
    var_9 = var_8.user()
    var_10 = '\\'
    var_11 = -1
    var_12 = user_path_win.split(var_10)[var_11]



# Parsed testcases at query #5
#--------------------------


def test_case_0():
    var_0 = '/'



