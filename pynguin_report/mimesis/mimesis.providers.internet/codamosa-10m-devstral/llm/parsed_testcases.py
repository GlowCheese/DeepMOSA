####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 'api'
    var_8 = 'www'
    var_9 = 'dev'
    var_10 = [var_7, var_8, var_9]
    var_11 = module_0.Internet()
    var_12 = var_11.url(subdomains=var_10)
    var_13 = module_0.Internet()
    var_14 = '.com'
    var_15 = '.org'
    var_16 = '.net'
    var_17 = (var_14, var_15, var_16)
    var_18 = module_0.Internet()
    var_19 = [var_7]
    var_20 = (var_14, var_15, var_16)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 5
    var_9 = var_7.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 2
    var_14 = var_12.slug(var_13)
    var_15 = module_1.split(var_2)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = module_0.Internet()
    var_18 = 12
    var_19 = var_17.slug(var_18)
    var_20 = module_1.split(var_2)
    var_21 = len(var_20)
    assert var_21 == 12
    var_22 = module_0.Internet()
    var_23 = 13
    var_24 = var_22.slug(var_23)
    var_25 = module_0.Internet()
    var_26 = 1
    var_27 = var_25.slug(var_26)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = 5
    var_8 = var_0.slug(var_7)
    var_9 = module_1.split(var_2)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 2
    var_12 = var_0.slug(var_11)
    var_13 = module_1.split(var_2)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 12
    var_16 = var_0.slug(var_15)
    var_17 = module_1.split(var_2)
    var_18 = len(var_17)
    assert var_18 == 12
    var_19 = 13
    var_20 = var_0.slug(var_19)
    var_21 = 1
    var_22 = var_0.slug(var_21)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'blog'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = 1
    var_5 = '//'
    var_6 = url.split(var_5)[var_4]
    var_7 = 'http://'
    var_8 = url_with_port.split(var_5)[var_4]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = [var_9, var_10]
    var_12 = var_0.url(subdomains=var_11)
    var_13 = [var_9, var_10]



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_0.Internet()
    var_4 = 3
    var_5 = var_3.slug(var_4)
    var_6 = module_0.Internet()
    var_7 = 12
    var_8 = var_6.slug(var_7)
    var_9 = module_0.Internet()
    var_10 = 13
    var_11 = var_9.slug(var_10)
    var_12 = module_0.Internet()
    var_13 = 1
    var_14 = var_12.slug(var_13)
    var_15 = module_0.Internet()
    var_16 = 5
    var_17 = var_15.slug(var_16)
    var_18 = module_1.split(var_14)
    var_19 = len(var_18)
    assert var_19 == 5



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '.'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    var_6 = '/'
    var_7 = 0
    var_8 = -1
    var_9 = ':'
    var_10 = url_with_port.split(var_9)[var_8]
    var_11 = var_12.split(var_6)[var_7]
    var_12 = int(var_11)
    var_13 = 'http://'
    var_14 = 'api'
    var_15 = 'dev'
    var_16 = [var_14, var_15]
    var_17 = var_0.url(subdomains=var_16)
    var_18 = [var_14, var_15]



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = -1
    var_8 = ':'
    var_9 = url.split(var_8)[var_7]
    var_10 = 'api'
    var_11 = 'cdn'
    var_12 = 'static'
    var_13 = [var_10, var_11, var_12]
    var_14 = module_0.Internet()
    var_15 = var_14.url(subdomains=var_13)
    var_16 = module_0.Internet()
    var_17 = '.com'
    var_18 = '.org'
    var_19 = '.net'
    var_20 = module_0.Internet()
    var_21 = [var_10]



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'dev'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = 0
    var_17 = 1
    var_18 = module_0.Internet()
    var_19 = [var_9]
    var_20 = url.split(var_3)[var_7]



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'blog'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = -1
    var_17 = '.'
    var_18 = url.split(var_17)[var_16]
    var_19 = module_0.Internet()
    var_20 = [var_9]
    var_21 = url.split(var_3)[var_7]
    var_22 = -1
    var_23 = url.split(var_17)[var_22]



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = 'api'
    var_5 = 'www'
    var_6 = [var_4, var_5]
    var_7 = var_0.url(subdomains=var_6)
    var_8 = [var_4, var_5]
    var_9 = 'http://'



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 1
    var_4 = var_2 >= var_3
    var_5 = len(var_1)
    var_6 = 10
    var_7 = var_5 <= var_6
    var_8 = module_0.Internet()
    var_9 = 5
    var_10 = var_8.query_parameters(var_9)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 32
    var_14 = var_12.query_parameters(var_13)
    var_15 = len(var_14)
    assert var_15 == 32
    var_16 = module_0.Internet()
    var_17 = 33
    var_18 = var_16.query_parameters(var_17)
    var_19 = module_0.Internet()
    var_20 = var_19.query_parameters(var_6)
    var_21 = len(var_20)
    var_22 = module_0.Internet()
    var_23 = var_22.query_parameters(var_9)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'static'
    var_11 = 'cdn'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = -1
    var_17 = '.'
    var_18 = url.split(var_17)[var_16]
    var_19 = module_0.Internet()
    var_20 = [var_9]
    var_21 = url.split(var_3)[var_7]
    var_22 = -1
    var_23 = url.split(var_17)[var_22]



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = module_0.Internet()
    var_4 = 5
    var_5 = var_3.query_parameters(var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = module_0.Internet()
    var_8 = 32
    var_9 = var_7.query_parameters(var_8)
    var_10 = len(var_9)
    assert var_10 == 32
    var_11 = module_0.Internet()
    var_12 = 33
    var_13 = var_11.query_parameters(var_12)
    var_14 = module_0.Internet()
    var_15 = 10
    var_16 = var_14.query_parameters(var_15)
    var_17 = len(var_16)
    var_18 = module_0.Internet()
    var_19 = 1
    var_20 = var_18.query_parameters(var_19)
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = 'http://'
    var_5 = 'api'
    var_6 = 'beta'
    var_7 = [var_5, var_6]
    var_8 = var_0.url(subdomains=var_7)
    var_9 = [var_5, var_6]



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_1.split(var_2)
    var_8 = 5
    var_9 = var_0.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = 13
    var_13 = var_0.slug(var_12)
    var_14 = 1
    var_15 = var_0.slug(var_14)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = 5
    var_8 = var_0.slug(var_7)
    var_9 = module_1.split(var_2)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 2
    var_12 = var_0.slug(var_11)
    var_13 = module_1.split(var_2)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 12
    var_16 = var_0.slug(var_15)
    var_17 = module_1.split(var_2)
    var_18 = len(var_17)
    assert var_18 == 12
    var_19 = 13
    var_20 = var_0.slug(var_19)
    var_21 = 1
    var_22 = var_0.slug(var_21)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = module_0.Internet()
    var_5 = 5
    var_6 = var_4.query_parameters(var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = module_0.Internet()
    var_9 = 32
    var_10 = var_8.query_parameters(var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = module_0.Internet()
    var_13 = 33
    var_14 = var_12.query_parameters(var_13)
    var_15 = module_0.Internet()
    var_16 = 10
    var_17 = var_15.query_parameters(var_16)
    var_18 = len(var_17)



# Parsed testcases at query #19
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = 5
    var_8 = var_0.slug(var_7)
    var_9 = module_1.split(var_2)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 2
    var_12 = var_0.slug(var_11)
    var_13 = module_1.split(var_2)
    var_14 = len(var_13)
    assert var_14 == 2
    var_15 = 12
    var_16 = var_0.slug(var_15)
    var_17 = module_1.split(var_2)
    var_18 = len(var_17)
    assert var_18 == 12
    var_19 = 1
    var_20 = var_0.slug(var_19)
    var_21 = 13
    var_22 = var_0.slug(var_21)



# Parsed testcases at query #20
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 5
    var_9 = var_7.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 13
    var_14 = var_12.slug(var_13)
    var_15 = module_0.Internet()
    var_16 = 1
    var_17 = var_15.slug(var_16)



####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 5
    var_9 = var_7.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 2
    var_14 = var_12.slug(var_13)
    var_15 = module_1.split(var_2)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = module_0.Internet()
    var_18 = 12
    var_19 = var_17.slug(var_18)
    var_20 = module_1.split(var_2)
    var_21 = len(var_20)
    assert var_21 == 12
    var_22 = module_0.Internet()
    var_23 = 13
    var_24 = var_22.slug(var_23)
    var_25 = module_0.Internet()
    var_26 = 1
    var_27 = var_25.slug(var_26)



# Parsed testcases at query #2
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 5
    var_9 = var_7.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 2
    var_14 = var_12.slug(var_13)
    var_15 = module_1.split(var_2)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = module_0.Internet()
    var_18 = 12
    var_19 = var_17.slug(var_18)
    var_20 = module_1.split(var_2)
    var_21 = len(var_20)
    assert var_21 == 12
    var_22 = module_0.Internet()
    var_23 = 1
    var_24 = var_22.slug(var_23)
    var_25 = module_0.Internet()
    var_26 = 13
    var_27 = var_25.slug(var_26)



# Parsed testcases at query #3
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = 5
    var_8 = var_0.slug(var_7)
    var_9 = module_1.split(var_2)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 1
    var_12 = var_0.slug(var_11)
    var_13 = 13
    var_14 = var_0.slug(var_13)



# Parsed testcases at query #4
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_0.Internet()
    var_4 = '-'
    var_5 = module_1.split(var_4)
    var_6 = module_0.Internet()
    var_7 = 1
    var_8 = var_6.slug(var_7)
    var_9 = module_0.Internet()
    var_10 = 13
    var_11 = var_9.slug(var_10)
    var_12 = module_0.Internet()
    var_13 = 3
    var_14 = var_12.slug(var_13)
    var_15 = module_0.Internet()
    var_16 = var_15.slug(var_13)



# Parsed testcases at query #5
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = 'sub'
    var_4 = 'domain'
    var_5 = [var_3, var_4]
    var_6 = var_0.url(subdomains=var_5)
    var_7 = [var_3, var_4]
    var_8 = 'http://'



# Parsed testcases at query #6
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'blog'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = -1
    var_17 = '.'
    var_18 = url.split(var_17)[var_16]
    var_19 = module_0.Internet()
    var_20 = [var_9]
    var_21 = url.split(var_3)[var_7]
    var_22 = -1
    var_23 = url.split(var_17)[var_22]



# Parsed testcases at query #7
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = 'http://'
    var_5 = 'api'
    var_6 = 'v1'
    var_7 = [var_5, var_6]
    var_8 = var_0.url(subdomains=var_7)
    var_9 = [var_5, var_6]



# Parsed testcases at query #8
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 1
    var_4 = var_2 >= var_3
    var_5 = len(var_1)
    var_6 = 10
    var_7 = var_5 <= var_6
    var_8 = module_0.Internet()
    var_9 = 5
    var_10 = var_8.query_parameters(var_9)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 32
    var_14 = var_12.query_parameters(var_13)
    var_15 = len(var_14)
    assert var_15 == 32
    var_16 = module_0.Internet()
    var_17 = 33
    var_18 = var_16.query_parameters(var_17)
    var_19 = module_0.Internet()
    var_20 = var_19.query_parameters(var_6)
    var_21 = len(var_20)



# Parsed testcases at query #9
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = module_0.Internet()
    var_4 = 5
    var_5 = var_3.query_parameters(var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = module_0.Internet()
    var_8 = 32
    var_9 = var_7.query_parameters(var_8)
    var_10 = len(var_9)
    assert var_10 == 32
    var_11 = module_0.Internet()
    var_12 = 33
    var_13 = var_11.query_parameters(var_12)
    var_14 = module_0.Internet()
    var_15 = 10
    var_16 = var_14.query_parameters(var_15)
    var_17 = len(var_16)
    var_18 = module_0.Internet()
    var_19 = 1
    var_20 = var_18.query_parameters(var_19)
    var_21 = len(var_20)
    assert var_21 == 1



# Parsed testcases at query #10
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_1.split(var_2)
    var_8 = 5
    var_9 = var_0.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = 1
    var_13 = var_0.slug(var_12)
    var_14 = 13
    var_15 = var_0.slug(var_14)



# Parsed testcases at query #11
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'beta'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()



# Parsed testcases at query #12
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 1
    var_4 = var_2 >= var_3
    var_5 = len(var_1)
    var_6 = 10
    var_7 = var_5 <= var_6
    var_8 = 5
    var_9 = var_0.query_parameters(var_8)
    var_10 = len(var_9)
    assert var_10 == 5
    var_11 = 32
    var_12 = var_0.query_parameters(var_11)
    var_13 = len(var_12)
    assert var_13 == 32
    var_14 = 33
    var_15 = var_0.query_parameters(var_14)
    var_16 = var_0.query_parameters(var_6)
    var_17 = len(var_16)
    var_18 = 3
    var_19 = var_0.query_parameters(var_18)



# Parsed testcases at query #13
#--------------------------


import mimesis.providers.internet as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.slug()
    var_2 = '-'
    var_3 = module_1.split(var_2)
    var_4 = len(var_3)
    var_5 = module_1.split(var_2)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 5
    var_9 = var_7.slug(var_8)
    var_10 = module_1.split(var_2)
    var_11 = len(var_10)
    assert var_11 == 5
    var_12 = module_0.Internet()
    var_13 = 2
    var_14 = var_12.slug(var_13)
    var_15 = module_1.split(var_2)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = module_0.Internet()
    var_18 = 12
    var_19 = var_17.slug(var_18)
    var_20 = module_1.split(var_2)
    var_21 = len(var_20)
    assert var_21 == 12
    var_22 = module_0.Internet()
    var_23 = 13
    var_24 = var_22.slug(var_23)
    var_25 = module_0.Internet()
    var_26 = 1
    var_27 = var_25.slug(var_26)



# Parsed testcases at query #14
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'cdn'
    var_11 = 'static'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = '.com'
    var_17 = '.org'
    var_18 = '.net'
    var_19 = (var_16, var_17, var_18)
    var_20 = module_0.Internet()
    var_21 = 'test'
    var_22 = [var_21]
    var_23 = url.split(var_3)[var_7]



# Parsed testcases at query #15
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.url()
    var_2 = 'https://'
    var_3 = '/'
    var_4 = module_0.Internet()
    var_5 = 'http://'
    var_6 = module_0.Internet()
    var_7 = 2
    var_8 = url.split(var_3)[var_7]
    var_9 = 'api'
    var_10 = 'www'
    var_11 = 'dev'
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.Internet()
    var_14 = var_13.url(subdomains=var_12)
    var_15 = module_0.Internet()
    var_16 = -1
    var_17 = '.'
    var_18 = url.split(var_17)[var_16]



# Parsed testcases at query #16
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = module_0.Internet()
    var_4 = 5
    var_5 = var_3.query_parameters(var_4)
    var_6 = len(var_5)
    assert var_6 == 5
    var_7 = module_0.Internet()
    var_8 = 32
    var_9 = var_7.query_parameters(var_8)
    var_10 = len(var_9)
    assert var_10 == 32
    var_11 = module_0.Internet()
    var_12 = 33
    var_13 = var_11.query_parameters(var_12)
    var_14 = module_0.Internet()
    var_15 = 10
    var_16 = var_14.query_parameters(var_15)



# Parsed testcases at query #17
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = len(var_1)
    var_4 = module_0.Internet()
    var_5 = 5
    var_6 = var_4.query_parameters(var_5)
    var_7 = len(var_6)
    assert var_7 == 5
    var_8 = module_0.Internet()
    var_9 = 32
    var_10 = var_8.query_parameters(var_9)
    var_11 = len(var_10)
    assert var_11 == 32
    var_12 = module_0.Internet()
    var_13 = 33
    var_14 = var_12.query_parameters(var_13)
    var_15 = module_0.Internet()
    var_16 = 3
    var_17 = var_15.query_parameters(var_16)
    var_18 = module_0.Internet()
    var_19 = 10
    var_20 = var_18.query_parameters(var_19)



# Parsed testcases at query #18
#--------------------------


import mimesis.providers.internet as module_0

def test_case_0():
    var_0 = module_0.Internet()
    var_1 = var_0.query_parameters()
    var_2 = len(var_1)
    var_3 = 5
    var_4 = module_0.Internet()
    var_5 = var_4.query_parameters(var_3)
    var_6 = len(var_5)
    var_7 = module_0.Internet()
    var_8 = 32
    var_9 = var_7.query_parameters(var_8)
    var_10 = len(var_9)
    assert var_10 == 32
    var_11 = module_0.Internet()
    var_12 = 33
    var_13 = var_11.query_parameters(var_12)



