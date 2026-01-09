####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.path as module_0


def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'linux'
    var_2 = module_0.Path(var_1)
    var_3 = 'unsupported'
    var_4 = module_0.Path(var_3)



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = 'Test method user of class Path.'
    var_1 = module_0.Path()
    var_2 = var_1.user()
    var_3 = var_1._pathlib_home
    var_4 = str(var_3)
    var_5 = -1
    var_6 = '/'
    var_7 = result.split(var_6)[var_5]
    var_8 = -1
    var_9 = '/'
    var_10 = result.split(var_9)[var_8]
    var_11 = -1
    var_12 = '/'
    var_13 = result.split(var_12)[var_11]



# Parsed testcases at query #3
#--------------------------


import pathlib as module_1


def test_case_0():
    var_0 = module_0.Path()
    var_1 = module_1.PurePosixPath()
    var_2 = 'home'



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0.user()
    var_2 = '/home/'
    var_3 = 'C:\\Users\\'
    var_4 = 'linux'
    var_5 = module_0.Path(var_4)
    var_6 = var_5.user()
    var_7 = 'win32'
    var_8 = module_0.Path(var_7)
    var_9 = var_8.user()
    var_10 = module_0.Path(var_7)
    var_11 = var_10.user()
    var_12 = -1
    var_13 = '\\'
    var_14 = result.split(var_13)[var_12]
    var_15 = 0
    var_16 = var_14[var_15]
    var_17 = module_0.Path(var_4)
    var_18 = var_17.user()
    var_19 = -1
    var_20 = '/'
    var_21 = result.split(var_20)[var_19]
    var_22 = module_0.Path(var_4)
    var_23 = var_22.user()
    var_24 = len(var_23)
    var_25 = len(var_2)
    var_26 = module_0.Path(var_4)
    var_27 = 10
    var_28 = range(var_27)



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '\\'
    var_4 = -1
    var_5 = user_path.split(var_3)[var_4]
    var_6 = 0
    var_7 = -1
    var_8 = user_path.split(var_3)[var_7][var_6]
    var_9 = 'linux'
    var_10 = module_0.Path(var_9)
    var_11 = var_10.user()
    var_12 = '/'
    var_13 = -1
    var_14 = user_path.split(var_12)[var_13]
    var_15 = -1
    var_16 = user_path.split(var_12)[var_15][var_6]
    var_17 = 'darwin'
    var_18 = module_0.Path(var_17)
    var_19 = var_18.user()
    var_20 = -1
    var_21 = user_path.split(var_12)[var_20]
    var_22 = -1
    var_23 = user_path.split(var_12)[var_22][var_6]
    var_24 = 'freebsd'
    var_25 = module_0.Path(var_24)
    var_26 = var_25.user()
    var_27 = -1
    var_28 = user_path.split(var_12)[var_27]
    var_29 = -1
    var_30 = user_path.split(var_12)[var_29][var_6]
    var_31 = 'win64'
    var_32 = module_0.Path(var_31)
    var_33 = var_32.user()
    var_34 = -1
    var_35 = user_path.split(var_3)[var_34]
    var_36 = -1
    var_37 = user_path.split(var_3)[var_36][var_6]



# Parsed testcases at query #6
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'linux'
    var_2 = module_0.Path(var_1)
    var_3 = 'win32'
    var_4 = module_0.Path(var_3)
    var_5 = 'freebsd'
    var_6 = module_0.Path(var_5)
    var_7 = 'win64'
    var_8 = module_0.Path(var_7)
    var_9 = 'darwin'
    var_10 = module_0.Path(var_9)
    var_11 = 'unknown'
    var_12 = module_0.Path(var_11)



# Parsed testcases at query #7
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = var_0._pathlib_home
    var_2 = var_0._pathlib_home
    var_3 = 'linux'
    var_4 = module_0.Path(var_3)
    var_5 = var_4._pathlib_home
    var_6 = 'win32'
    var_7 = module_0.Path(var_6)
    var_8 = var_7._pathlib_home
    var_9 = 'freebsd'
    var_10 = module_0.Path(var_9)
    var_11 = var_10._pathlib_home



# Parsed testcases at query #8
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'win32'
    var_2 = module_0.Path(var_1)
    var_3 = 'freebsd'
    var_4 = module_0.Path(var_3)



# Parsed testcases at query #9
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = module_1.PurePosixPath()
    var_2 = 'home'



# Parsed testcases at query #10
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'win'
    var_2 = var_0._pathlib_home
    var_3 = var_0._pathlib_home
    var_4 = 'win32'
    var_5 = module_0.Path(var_4)
    var_6 = var_5._pathlib_home
    var_7 = 'freebsd'
    var_8 = module_0.Path(var_7)
    var_9 = var_8._pathlib_home



####################################################################
#     TEST GENERATION BEGINS (CODAMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------



def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '\\'
    var_4 = -1
    var_5 = result.split(var_3)[var_4]
    var_6 = -1
    var_7 = result.split(var_3)[var_6]
    var_8 = 'linux'
    var_9 = module_0.Path(var_8)
    var_10 = var_9.user()
    var_11 = '/'
    var_12 = -1
    var_13 = result.split(var_11)[var_12]
    var_14 = -1
    var_15 = result.split(var_11)[var_14]
    var_16 = 'darwin'
    var_17 = module_0.Path(var_16)
    var_18 = var_17.user()
    var_19 = -1
    var_20 = result.split(var_11)[var_19]
    var_21 = -1
    var_22 = result.split(var_11)[var_21]
    var_23 = 'freebsd'
    var_24 = module_0.Path(var_23)
    var_25 = var_24.user()
    var_26 = -1
    var_27 = result.split(var_11)[var_26]
    var_28 = -1
    var_29 = result.split(var_11)[var_28]
    var_30 = 'win64'
    var_31 = module_0.Path(var_30)
    var_32 = var_31.user()
    var_33 = -1
    var_34 = result.split(var_3)[var_33]
    var_35 = -1
    var_36 = result.split(var_3)[var_35]
    var_37 = 'unknown'
    var_38 = module_0.Path(var_37)
    var_39 = var_38.user()
    var_40 = -1
    var_41 = result.split(var_11)[var_40]
    var_42 = -1
    var_43 = result.split(var_11)[var_42]



# Parsed testcases at query #2
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'win32'
    var_2 = module_0.Path(var_1)
    var_3 = 'freebsd'
    var_4 = module_0.Path(var_3)
    var_5 = 'unsupported'
    var_6 = module_0.Path(var_5)



# Parsed testcases at query #3
#--------------------------



def test_case_0():
    var_0 = module_0.Path()
    var_1 = 'linux'
    var_2 = module_0.Path(var_1)
    var_3 = 'freebsd'
    var_4 = module_0.Path(var_3)
    var_5 = 'win32'
    var_6 = module_0.Path(var_5)
    var_7 = 'win64'
    var_8 = module_0.Path(var_7)
    var_9 = 'darwin'
    var_10 = module_0.Path(var_9)
    var_11 = 'invalid'
    var_12 = module_0.Path(var_11)



# Parsed testcases at query #4
#--------------------------



def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '\\'
    var_4 = -1
    var_5 = user_path.split(var_3)[var_4]
    var_6 = 'linux'
    var_7 = module_0.Path(var_6)
    var_8 = var_7.user()
    var_9 = '/'
    var_10 = -1
    var_11 = user_path.split(var_9)[var_10]
    var_12 = 'darwin'
    var_13 = module_0.Path(var_12)
    var_14 = var_13.user()
    var_15 = -1
    var_16 = user_path.split(var_9)[var_15]
    var_17 = 'freebsd'
    var_18 = module_0.Path(var_17)
    var_19 = var_18.user()
    var_20 = -1
    var_21 = user_path.split(var_9)[var_20]
    var_22 = module_0.Path()
    var_23 = var_22.user()
    var_24 = -1
    var_25 = user_path.split(var_9)[var_24]
    var_26 = -1
    var_27 = user_path.split(var_3)[var_26]



# Parsed testcases at query #5
#--------------------------



def test_case_0():
    var_0 = 'win32'
    var_1 = module_0.Path(var_0)
    var_2 = var_1.user()
    var_3 = '\\home\\'
    var_4 = '\\'
    var_5 = 2
    var_6 = result.split(var_4)[var_5]
    var_7 = 'linux'
    var_8 = module_0.Path(var_7)
    var_9 = var_8.user()
    var_10 = '/home/'
    var_11 = '/'
    var_12 = result.split(var_11)[var_5]
    var_13 = 'darwin'
    var_14 = module_0.Path(var_13)
    var_15 = var_14.user()
    var_16 = result.split(var_11)[var_5]
    var_17 = 'freebsd'
    var_18 = module_0.Path(var_17)
    var_19 = var_18.user()
    var_20 = result.split(var_11)[var_5]
    var_21 = 'unknown'
    var_22 = module_0.Path(var_21)
    var_23 = var_22.user()
    var_24 = result.split(var_11)[var_5]



