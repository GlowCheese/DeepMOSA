####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_visit_function_def_node. Retrieved 7/9 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 7/9 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'body'
    var_7 = 'type_ignores'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Module(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.With(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.AsyncWith(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_11, var_16: var_12, var_17: var_13}
    var_19 = module_1.While(*var_14, **var_18)
    var_20 = var_2.visit(var_19)
    var_21 = bool(var_19 not in var_2._no_fall_through_nodes)
    assert var_21 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Store(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_11, var_25: var_20, var_26: var_21, var_27: var_22}
    var_29 = module_1.For(*var_23, **var_28)
    var_30 = var_2.visit(var_29)
    var_31 = bool(var_29 not in var_2._no_fall_through_nodes)
    assert var_31 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Store(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_11, var_25: var_20, var_26: var_21, var_27: var_22}
    var_29 = module_1.AsyncFor(*var_23, **var_28)
    var_30 = var_2.visit(var_29)
    var_31 = bool(var_29 not in var_2._no_fall_through_nodes)
    assert var_31 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_11, var_16: var_12, var_17: var_13}
    var_19 = module_1.If(*var_14, **var_18)
    var_20 = var_2.visit(var_19)
    var_21 = bool(var_19 not in var_2._no_fall_through_nodes)
    assert var_21 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = 'z'
    var_22 = []
    var_23 = {}
    var_24 = module_1.Load(*var_22, **var_23)
    var_25 = []
    var_26 = 'id'
    var_27 = 'ctx'
    var_28 = {var_26: var_21, var_27: var_24}
    var_29 = module_1.Name(*var_25, **var_28)
    var_30 = []
    var_31 = 'test'
    var_32 = 'body'
    var_33 = 'orelse'
    var_34 = {var_31: var_11, var_32: var_20, var_33: var_29}
    var_35 = module_1.IfExp(*var_30, **var_34)
    var_36 = var_2.visit(var_35)
    var_37 = bool(var_35 not in var_2._no_fall_through_nodes)
    assert var_37 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_1.Try(*var_7, **var_12)
    var_14 = var_2.visit(var_13)
    var_15 = bool(var_13 not in var_2._no_fall_through_nodes)
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #11
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_visit_break_node. Retrieved 1/4 statements.
# Partially parsed test_visit_continue_node. Retrieved 1/4 statements.
# Partially parsed test_visit_return_node. Retrieved 1/4 statements.
# Partially parsed test_visit_raise_node. Retrieved 1/4 statements.
# Partially parsed test_visit_module_node. Retrieved 4/8 statements.
# Partially parsed test_visit_function_def_node. Retrieved 5/10 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 5/10 statements.
# Partially parsed test_visit_with_node. Retrieved 5/9 statements.
# Partially parsed test_visit_async_with_node. Retrieved 5/9 statements.
# Partially parsed test_visit_while_node. Retrieved 7/11 statements.
# Partially parsed test_visit_for_node. Retrieved 10/14 statements.
# Partially parsed test_visit_async_for_node. Retrieved 10/14 statements.
# Partially parsed test_visit_if_node. Retrieved 7/11 statements.
# Partially parsed test_visit_if_exp_node. Retrieved 10/14 statements.
# Partially parsed test_visit_try_node. Retrieved 6/10 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Break(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Continue(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Return(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Raise(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Pass(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = 'body'
    var_7 = 'type_ignores'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_0.Module(*var_5, **var_8)

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.arguments(*var_1, **var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_0.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.arguments(*var_1, **var_2)
    var_4 = []
    var_5 = {}
    var_6 = module_0.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = []

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.withitem(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = 'items'
    var_10 = 'body'
    var_11 = {var_9: var_3, var_10: var_7}
    var_12 = module_0.With(*var_8, **var_11)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.withitem(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = 'items'
    var_10 = 'body'
    var_11 = {var_9: var_3, var_10: var_7}
    var_12 = module_0.AsyncWith(*var_8, **var_11)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = {}
    var_11 = module_0.Pass(*var_9, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_8, var_16: var_12, var_17: var_13}
    var_19 = module_0.While(*var_14, **var_18)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = []
    var_19 = {}
    var_20 = module_0.Pass(*var_18, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_8, var_25: var_17, var_26: var_21, var_27: var_22}
    var_29 = module_0.For(*var_23, **var_28)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = []
    var_19 = {}
    var_20 = module_0.Pass(*var_18, **var_19)
    var_21 = [var_20]
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_8, var_25: var_17, var_26: var_21, var_27: var_22}
    var_29 = module_0.AsyncFor(*var_23, **var_28)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = {}
    var_11 = module_0.Pass(*var_9, **var_10)
    var_12 = [var_11]
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_8, var_16: var_12, var_17: var_13}
    var_19 = module_0.If(*var_14, **var_18)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = 'z'
    var_19 = []
    var_20 = {}
    var_21 = module_0.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_18, var_24: var_21}
    var_26 = module_0.Name(*var_22, **var_25)
    var_27 = []
    var_28 = 'test'
    var_29 = 'body'
    var_30 = 'orelse'
    var_31 = {var_28: var_8, var_29: var_17, var_30: var_26}
    var_32 = module_0.IfExp(*var_27, **var_31)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Pass(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_0.Try(*var_7, **var_12)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_with_function_def_node. Retrieved 7/9 statements.
# Partially parsed test_visit_with_async_function_def_node. Retrieved 7/9 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'body'
    var_7 = 'type_ignores'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Module(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.With(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.AsyncWith(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = 'orelse'
    var_14 = {var_11: var_7, var_12: var_8, var_13: var_9}
    var_15 = module_1.While(*var_10, **var_14)
    var_16 = var_2.visit(var_15)
    var_17 = bool(var_15 in var_2._no_fall_through_nodes)
    assert var_17 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'target'
    var_17 = 'iter'
    var_18 = 'body'
    var_19 = 'orelse'
    var_20 = {var_16: var_7, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.For(*var_15, **var_20)
    var_22 = var_2.visit(var_21)
    var_23 = bool(var_21 not in var_2._no_fall_through_nodes)
    assert var_23 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'target'
    var_17 = 'iter'
    var_18 = 'body'
    var_19 = 'orelse'
    var_20 = {var_16: var_7, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.AsyncFor(*var_15, **var_20)
    var_22 = var_2.visit(var_21)
    var_23 = bool(var_21 not in var_2._no_fall_through_nodes)
    assert var_23 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = 'orelse'
    var_14 = {var_11: var_7, var_12: var_8, var_13: var_9}
    var_15 = module_1.If(*var_10, **var_14)
    var_16 = var_2.visit(var_15)
    var_17 = bool(var_15 not in var_2._no_fall_through_nodes)
    assert var_17 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = 'value'
    var_10 = {var_9: var_3}
    var_11 = module_1.NameConstant(*var_8, **var_10)
    var_12 = False
    var_13 = []
    var_14 = 'value'
    var_15 = {var_14: var_12}
    var_16 = module_1.NameConstant(*var_13, **var_15)
    var_17 = []
    var_18 = 'test'
    var_19 = 'body'
    var_20 = 'orelse'
    var_21 = {var_18: var_7, var_19: var_11, var_20: var_16}
    var_22 = module_1.IfExp(*var_17, **var_21)
    var_23 = var_2.visit(var_22)
    var_24 = bool(var_22 not in var_2._no_fall_through_nodes)
    assert var_24 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_1.Try(*var_7, **var_12)
    var_14 = var_2.visit(var_13)
    var_15 = bool(var_13 not in var_2._no_fall_through_nodes)
    assert var_15 is True



# Parsed testcases at query #3
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #18
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #23
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #24
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #26
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #37
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #40
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #41
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #45
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Reachability(var_0)
    var_2 = bool(not var_1._no_fall_through_nodes)
    assert var_2 is True



# Parsed testcases at query #47
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes_as_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #49
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #50
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #51
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #54
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #55
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #59
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #61
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #62
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #63
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #64
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #65
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #66
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #67
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #68
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_reachability_initialization. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #70
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #71
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #72
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #73
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #74
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #75
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #76
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_reachability_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = set()



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_init_no_fall_through_nodes_is_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #79
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #80
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #81
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #82
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #83
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #84
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #85
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #86
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #87
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #89
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #90
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #91
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #92
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #94
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #95
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #96
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #97
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #98
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #99
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #100
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #101
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #102
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #103
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #105
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #106
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_break_node. Retrieved 1/4 statements.
# Partially parsed test_visit_continue_node. Retrieved 1/4 statements.
# Partially parsed test_visit_return_node. Retrieved 1/4 statements.
# Partially parsed test_visit_raise_node. Retrieved 1/4 statements.
# Partially parsed test_visit_module_node. Retrieved 3/7 statements.
# Partially parsed test_visit_function_def_node. Retrieved 4/9 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 4/9 statements.
# Partially parsed test_visit_with_node. Retrieved 4/8 statements.
# Partially parsed test_visit_async_with_node. Retrieved 4/8 statements.
# Partially parsed test_visit_while_node. Retrieved 5/9 statements.
# Partially parsed test_visit_for_node. Retrieved 7/11 statements.
# Partially parsed test_visit_async_for_node. Retrieved 7/11 statements.
# Partially parsed test_visit_if_node. Retrieved 6/10 statements.
# Partially parsed test_visit_if_exp_node. Retrieved 7/11 statements.
# Partially parsed test_visit_try_node. Retrieved 6/10 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Break(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Continue(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Return(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Raise(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Pass(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = 'body'
    var_6 = {var_5: var_3}
    var_7 = module_0.Module(*var_4, **var_6)

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Pass(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = {}
    var_7 = module_0.arguments(*var_5, **var_6)
    var_8 = []

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Pass(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = {}
    var_7 = module_0.arguments(*var_5, **var_6)
    var_8 = []

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = {}
    var_3 = module_0.Pass(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_0, var_7: var_4}
    var_9 = module_0.With(*var_5, **var_8)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = {}
    var_3 = module_0.Pass(*var_1, **var_2)
    var_4 = [var_3]
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_0, var_7: var_4}
    var_9 = module_0.AsyncWith(*var_5, **var_8)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.NameConstant(*var_1, **var_3)
    var_5 = []
    var_6 = {}
    var_7 = module_0.Pass(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = 'test'
    var_11 = 'body'
    var_12 = {var_10: var_4, var_11: var_8}
    var_13 = module_0.While(*var_9, **var_12)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'y'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = []
    var_11 = {}
    var_12 = module_0.Pass(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'target'
    var_16 = 'iter'
    var_17 = 'body'
    var_18 = {var_15: var_4, var_16: var_9, var_17: var_13}
    var_19 = module_0.For(*var_14, **var_18)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'id'
    var_3 = {var_2: var_0}
    var_4 = module_0.Name(*var_1, **var_3)
    var_5 = 'y'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = []
    var_11 = {}
    var_12 = module_0.Pass(*var_10, **var_11)
    var_13 = [var_12]
    var_14 = []
    var_15 = 'target'
    var_16 = 'iter'
    var_17 = 'body'
    var_18 = {var_15: var_4, var_16: var_9, var_17: var_13}
    var_19 = module_0.AsyncFor(*var_14, **var_18)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.NameConstant(*var_1, **var_3)
    var_5 = []
    var_6 = {}
    var_7 = module_0.Pass(*var_5, **var_6)
    var_8 = [var_7]
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = 'orelse'
    var_14 = {var_11: var_4, var_12: var_8, var_13: var_9}
    var_15 = module_0.If(*var_10, **var_14)

import ast as module_0

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'value'
    var_3 = {var_2: var_0}
    var_4 = module_0.NameConstant(*var_1, **var_3)
    var_5 = 'x'
    var_6 = []
    var_7 = 'id'
    var_8 = {var_7: var_5}
    var_9 = module_0.Name(*var_6, **var_8)
    var_10 = 'y'
    var_11 = []
    var_12 = 'id'
    var_13 = {var_12: var_10}
    var_14 = module_0.Name(*var_11, **var_13)
    var_15 = []
    var_16 = 'test'
    var_17 = 'body'
    var_18 = 'orelse'
    var_19 = {var_16: var_4, var_17: var_9, var_18: var_14}
    var_20 = module_0.IfExp(*var_15, **var_19)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Pass(*var_0, **var_1)
    var_3 = [var_2]
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_0.Try(*var_7, **var_12)



# Parsed testcases at query #3
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'test'
    var_15 = 'body'
    var_16 = 'orelse'
    var_17 = {var_14: var_7, var_15: var_11, var_16: var_12}
    var_18 = module_1.While(*var_13, **var_17)
    var_19 = var_2.visit(var_18)
    var_20 = bool(var_18 in var_2._no_fall_through_nodes)
    assert var_20 is True



# Parsed testcases at query #5
#--------------------------




import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'target'
    var_17 = 'iter'
    var_18 = 'body'
    var_19 = 'orelse'
    var_20 = {var_16: var_7, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.For(*var_15, **var_20)
    var_22 = var_2.visit(var_21)
    var_23 = bool(var_21 in var_2._no_fall_through_nodes or True)
    assert var_23 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_visit_break_node. Retrieved 1/4 statements.
# Partially parsed test_visit_continue_node. Retrieved 1/4 statements.
# Partially parsed test_visit_return_node. Retrieved 1/4 statements.
# Partially parsed test_visit_raise_node. Retrieved 1/4 statements.
# Partially parsed test_visit_module_node. Retrieved 3/6 statements.
# Partially parsed test_visit_function_def_node. Retrieved 4/8 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 4/8 statements.
# Partially parsed test_visit_with_node. Retrieved 3/6 statements.
# Partially parsed test_visit_async_with_node. Retrieved 3/6 statements.
# Partially parsed test_visit_while_node. Retrieved 5/8 statements.
# Partially parsed test_visit_for_node. Retrieved 9/12 statements.
# Partially parsed test_visit_async_for_node. Retrieved 9/12 statements.
# Partially parsed test_visit_if_node. Retrieved 6/9 statements.
# Partially parsed test_visit_if_exp_node. Retrieved 10/13 statements.
# Partially parsed test_visit_try_node. Retrieved 5/8 statements.


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Break(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Continue(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Return(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.Raise(*var_0, **var_1)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'body'
    var_4 = 'type_ignores'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.Module(*var_2, **var_5)

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.arguments(*var_1, **var_2)
    var_4 = []
    var_5 = []
    var_6 = []

import ast as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = {}
    var_3 = module_0.arguments(*var_1, **var_2)
    var_4 = []
    var_5 = []
    var_6 = []

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'items'
    var_4 = 'body'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.With(*var_2, **var_5)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = 'items'
    var_4 = 'body'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.AsyncWith(*var_2, **var_5)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = {var_11: var_8, var_12: var_9}
    var_14 = module_0.While(*var_10, **var_13)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = 'target'
    var_22 = 'iter'
    var_23 = 'body'
    var_24 = 'orelse'
    var_25 = {var_21: var_8, var_22: var_17, var_23: var_18, var_24: var_19}
    var_26 = module_0.For(*var_20, **var_25)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Store(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = 'target'
    var_22 = 'iter'
    var_23 = 'body'
    var_24 = 'orelse'
    var_25 = {var_21: var_8, var_22: var_17, var_23: var_18, var_24: var_19}
    var_26 = module_0.AsyncFor(*var_20, **var_25)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = []
    var_10 = []
    var_11 = []
    var_12 = 'test'
    var_13 = 'body'
    var_14 = 'orelse'
    var_15 = {var_12: var_8, var_13: var_9, var_14: var_10}
    var_16 = module_0.If(*var_11, **var_15)

import ast as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = {}
    var_3 = module_0.Load(*var_1, **var_2)
    var_4 = []
    var_5 = 'id'
    var_6 = 'ctx'
    var_7 = {var_5: var_0, var_6: var_3}
    var_8 = module_0.Name(*var_4, **var_7)
    var_9 = 'y'
    var_10 = []
    var_11 = {}
    var_12 = module_0.Load(*var_10, **var_11)
    var_13 = []
    var_14 = 'id'
    var_15 = 'ctx'
    var_16 = {var_14: var_9, var_15: var_12}
    var_17 = module_0.Name(*var_13, **var_16)
    var_18 = 'z'
    var_19 = []
    var_20 = {}
    var_21 = module_0.Load(*var_19, **var_20)
    var_22 = []
    var_23 = 'id'
    var_24 = 'ctx'
    var_25 = {var_23: var_18, var_24: var_21}
    var_26 = module_0.Name(*var_22, **var_25)
    var_27 = []
    var_28 = 'test'
    var_29 = 'body'
    var_30 = 'orelse'
    var_31 = {var_28: var_8, var_29: var_17, var_30: var_26}
    var_32 = module_0.IfExp(*var_27, **var_31)

import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []
    var_3 = []
    var_4 = []
    var_5 = 'body'
    var_6 = 'handlers'
    var_7 = 'orelse'
    var_8 = 'finalbody'
    var_9 = {var_5: var_0, var_6: var_1, var_7: var_2, var_8: var_3}
    var_10 = module_0.Try(*var_4, **var_9)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_visit_function_def_node. Retrieved 7/9 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 7/9 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'body'
    var_7 = 'type_ignores'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Module(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.With(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.AsyncWith(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = 'orelse'
    var_14 = {var_11: var_7, var_12: var_8, var_13: var_9}
    var_15 = module_1.While(*var_10, **var_14)
    var_16 = var_2.visit(var_15)
    var_17 = bool(var_15 not in var_2._no_fall_through_nodes)
    assert var_17 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'target'
    var_17 = 'iter'
    var_18 = 'body'
    var_19 = 'orelse'
    var_20 = {var_16: var_7, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.For(*var_15, **var_20)
    var_22 = var_2.visit(var_21)
    var_23 = bool(var_21 not in var_2._no_fall_through_nodes)
    assert var_23 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = []
    var_15 = []
    var_16 = 'target'
    var_17 = 'iter'
    var_18 = 'body'
    var_19 = 'orelse'
    var_20 = {var_16: var_7, var_17: var_12, var_18: var_13, var_19: var_14}
    var_21 = module_1.AsyncFor(*var_15, **var_20)
    var_22 = var_2.visit(var_21)
    var_23 = bool(var_21 not in var_2._no_fall_through_nodes)
    assert var_23 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'test'
    var_12 = 'body'
    var_13 = 'orelse'
    var_14 = {var_11: var_7, var_12: var_8, var_13: var_9}
    var_15 = module_1.If(*var_10, **var_14)
    var_16 = var_2.visit(var_15)
    var_17 = bool(var_15 not in var_2._no_fall_through_nodes)
    assert var_17 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = 'z'
    var_14 = []
    var_15 = 'id'
    var_16 = {var_15: var_13}
    var_17 = module_1.Name(*var_14, **var_16)
    var_18 = []
    var_19 = 'test'
    var_20 = 'body'
    var_21 = 'orelse'
    var_22 = {var_19: var_7, var_20: var_12, var_21: var_17}
    var_23 = module_1.IfExp(*var_18, **var_22)
    var_24 = var_2.visit(var_23)
    var_25 = bool(var_23 not in var_2._no_fall_through_nodes)
    assert var_25 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_1.Try(*var_7, **var_12)
    var_14 = var_2.visit(var_13)
    var_15 = bool(var_13 not in var_2._no_fall_through_nodes)
    assert var_15 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #15
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #16
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #20
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #22
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #23
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_no_fall_through_nodes_initialization.




# Parsed testcases at query #25
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #27
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #32
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #33
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #35
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #36
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #37
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #38
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #39
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #40
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #45
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #47
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #48
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #50
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #51
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #52
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_init_initializes_no_fall_through_nodes_as_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #55
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #57
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #58
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #59
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #61
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #62
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #63
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #64
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #66
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_reachability_initialization. Retrieved 1/3 statements.


def test_case_0():
    var_0 = set()



# Parsed testcases at query #68
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #69
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #70
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #71
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #72
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #73
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_reachability_initialization. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #75
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #76
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #77
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #78
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #79
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #80
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #81
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #82
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #83
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #84
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #85
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #87
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_init_initializes_no_fall_through_nodes_as_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #89
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #90
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #91
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #92
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #94
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #95
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #96
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #97
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #98
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #99
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #100
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #101
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #102
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #103
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #104
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #105
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #106
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #107
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #109
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #110
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #112
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #113
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #114
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #115
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #116
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #117
#--------------------------

# Partially parsed test_init_initializes_no_fall_through_nodes_as_set. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #118
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #119
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #120
#--------------------------

# Partially parsed test_init_creates_no_fall_through_nodes_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #121
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #122
#--------------------------

# Failed to parse test_init_no_fall_through_nodes_is_empty.




# Parsed testcases at query #123
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #124
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #125
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #126
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #127
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #128
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #129
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #130
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_visit_with_function_def_node. Retrieved 7/9 statements.
# Partially parsed test_visit_with_async_function_def_node. Retrieved 7/9 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'body'
    var_7 = 'type_ignores'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.Module(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = []
    var_9 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.With(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = 'items'
    var_7 = 'body'
    var_8 = {var_6: var_3, var_7: var_4}
    var_9 = module_1.AsyncWith(*var_5, **var_8)
    var_10 = var_2.visit(var_9)
    var_11 = bool(var_9 not in var_2._no_fall_through_nodes)
    assert var_11 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_11, var_16: var_12, var_17: var_13}
    var_19 = module_1.While(*var_14, **var_18)
    var_20 = var_2.visit(var_19)
    var_21 = bool(var_19 not in var_2._no_fall_through_nodes)
    assert var_21 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Store(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_11, var_25: var_20, var_26: var_21, var_27: var_22}
    var_29 = module_1.For(*var_23, **var_28)
    var_30 = var_2.visit(var_29)
    var_31 = bool(var_29 not in var_2._no_fall_through_nodes)
    assert var_31 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Store(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = []
    var_22 = []
    var_23 = []
    var_24 = 'target'
    var_25 = 'iter'
    var_26 = 'body'
    var_27 = 'orelse'
    var_28 = {var_24: var_11, var_25: var_20, var_26: var_21, var_27: var_22}
    var_29 = module_1.AsyncFor(*var_23, **var_28)
    var_30 = var_2.visit(var_29)
    var_31 = bool(var_29 not in var_2._no_fall_through_nodes)
    assert var_31 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = 'test'
    var_16 = 'body'
    var_17 = 'orelse'
    var_18 = {var_15: var_11, var_16: var_12, var_17: var_13}
    var_19 = module_1.If(*var_14, **var_18)
    var_20 = var_2.visit(var_19)
    var_21 = bool(var_19 not in var_2._no_fall_through_nodes)
    assert var_21 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Load(*var_4, **var_5)
    var_7 = []
    var_8 = 'id'
    var_9 = 'ctx'
    var_10 = {var_8: var_3, var_9: var_6}
    var_11 = module_1.Name(*var_7, **var_10)
    var_12 = 'y'
    var_13 = []
    var_14 = {}
    var_15 = module_1.Load(*var_13, **var_14)
    var_16 = []
    var_17 = 'id'
    var_18 = 'ctx'
    var_19 = {var_17: var_12, var_18: var_15}
    var_20 = module_1.Name(*var_16, **var_19)
    var_21 = 'z'
    var_22 = []
    var_23 = {}
    var_24 = module_1.Load(*var_22, **var_23)
    var_25 = []
    var_26 = 'id'
    var_27 = 'ctx'
    var_28 = {var_26: var_21, var_27: var_24}
    var_29 = module_1.Name(*var_25, **var_28)
    var_30 = []
    var_31 = 'test'
    var_32 = 'body'
    var_33 = 'orelse'
    var_34 = {var_31: var_11, var_32: var_20, var_33: var_29}
    var_35 = module_1.IfExp(*var_30, **var_34)
    var_36 = var_2.visit(var_35)
    var_37 = bool(var_35 not in var_2._no_fall_through_nodes)
    assert var_37 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = 'body'
    var_9 = 'handlers'
    var_10 = 'orelse'
    var_11 = 'finalbody'
    var_12 = {var_8: var_3, var_9: var_4, var_10: var_5, var_11: var_6}
    var_13 = module_1.Try(*var_7, **var_12)
    var_14 = var_2.visit(var_13)
    var_15 = bool(var_13 not in var_2._no_fall_through_nodes)
    assert var_15 is True



# Parsed testcases at query #2
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #3
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #5
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #7
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #11
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #20
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #24
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #25
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #31
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #32
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #33
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #34
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #37
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #38
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #39
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True



# Parsed testcases at query #40
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #41
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #42
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #44
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #46
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #47
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #48
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #49
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #50
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #51
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #52
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #54
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #55
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #56
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #57
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #58
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Reachability(var_0)
    var_2 = set()
    var_3 = var_1._no_fall_through_nodes
    var_4 = bool(var_1._no_fall_through_nodes == var_2)
    assert var_4 is True



# Parsed testcases at query #59
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #60
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #61
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #62
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #63
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #64
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #65
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #66
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #67
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #68
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #69
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #70
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #71
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #72
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_init_creates_no_fall_through_nodes_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #74
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #75
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #76
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #77
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #78
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #79
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #80
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Reachability(var_0)
    var_2 = bool(not var_1._no_fall_through_nodes)
    assert var_2 is True



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #82
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #83
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #84
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #85
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Reachability(var_0)
    var_2 = set()
    var_3 = var_1._no_fall_through_nodes
    var_4 = bool(var_1._no_fall_through_nodes == var_2)
    assert var_4 is True



# Parsed testcases at query #86
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #87
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #89
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #90
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #91
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #92
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #94
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #95
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #96
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #97
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #98
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #99
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #100
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_with_module_node. Retrieved 8/9 statements.
# Partially parsed test_visit_with_function_def_node. Retrieved 8/11 statements.
# Partially parsed test_visit_with_async_function_def_node. Retrieved 8/11 statements.
# Partially parsed test_visit_with_while_node. Retrieved 10/11 statements.
# Partially parsed test_visit_with_for_node. Retrieved 12/13 statements.
# Partially parsed test_visit_with_async_for_node. Retrieved 12/13 statements.
# Partially parsed test_visit_with_if_node. Retrieved 10/11 statements.
# Partially parsed test_visit_with_if_exp_node. Retrieved 11/12 statements.
# Partially parsed test_visit_with_try_node. Retrieved 10/11 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Pass(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = 'body'
    var_10 = 'type_ignores'
    var_11 = {var_9: var_6, var_10: var_7}
    var_12 = module_1.Module(*var_8, **var_11)
    var_13 = var_2.visit(var_12)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_1.Pass(*var_7, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.arguments(*var_4, **var_5)
    var_7 = []
    var_8 = {}
    var_9 = module_1.Pass(*var_7, **var_8)
    var_10 = [var_9]
    var_11 = []
    var_12 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'test'
    var_15 = 'body'
    var_16 = 'orelse'
    var_17 = {var_14: var_7, var_15: var_11, var_16: var_12}
    var_18 = module_1.While(*var_13, **var_17)
    var_19 = var_2.visit(var_18)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = 'target'
    var_20 = 'iter'
    var_21 = 'body'
    var_22 = 'orelse'
    var_23 = {var_19: var_7, var_20: var_12, var_21: var_16, var_22: var_17}
    var_24 = module_1.For(*var_18, **var_23)
    var_25 = var_2.visit(var_24)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = 'target'
    var_20 = 'iter'
    var_21 = 'body'
    var_22 = 'orelse'
    var_23 = {var_19: var_7, var_20: var_12, var_21: var_16, var_22: var_17}
    var_24 = module_1.AsyncFor(*var_18, **var_23)
    var_25 = var_2.visit(var_24)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'test'
    var_15 = 'body'
    var_16 = 'orelse'
    var_17 = {var_14: var_7, var_15: var_11, var_16: var_12}
    var_18 = module_1.If(*var_13, **var_17)
    var_19 = var_2.visit(var_18)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = 'x'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = 'y'
    var_14 = []
    var_15 = 'id'
    var_16 = {var_15: var_13}
    var_17 = module_1.Name(*var_14, **var_16)
    var_18 = []
    var_19 = 'test'
    var_20 = 'body'
    var_21 = 'orelse'
    var_22 = {var_19: var_7, var_20: var_12, var_21: var_17}
    var_23 = module_1.IfExp(*var_18, **var_22)
    var_24 = var_2.visit(var_23)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Pass(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'body'
    var_12 = 'handlers'
    var_13 = 'orelse'
    var_14 = 'finalbody'
    var_15 = {var_11: var_6, var_12: var_7, var_13: var_8, var_14: var_9}
    var_16 = module_1.Try(*var_10, **var_15)
    var_17 = var_2.visit(var_16)



# Parsed testcases at query #3
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #7
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #10
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #15
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #18
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #22
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Devstral t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_visit_function_def_node. Retrieved 7/9 statements.
# Partially parsed test_visit_async_function_def_node. Retrieved 7/9 statements.


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Break(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Continue(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Return(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Raise(*var_3, **var_4)
    var_6 = var_2.visit(var_5)
    var_7 = bool(var_5 in var_2._no_fall_through_nodes)
    assert var_7 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Pass(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = 'body'
    var_9 = {var_8: var_6}
    var_10 = module_1.Module(*var_7, **var_9)
    var_11 = var_2.visit(var_10)
    var_12 = bool(var_10 not in var_2._no_fall_through_nodes)
    assert var_12 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = {}
    var_10 = module_1.arguments(*var_8, **var_9)
    var_11 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'test'
    var_4 = []
    var_5 = {}
    var_6 = module_1.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = {}
    var_10 = module_1.arguments(*var_8, **var_9)
    var_11 = []

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_1.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = 'items'
    var_10 = 'body'
    var_11 = {var_9: var_3, var_10: var_7}
    var_12 = module_1.With(*var_8, **var_11)
    var_13 = var_2.visit(var_12)
    var_14 = bool(var_12 not in var_2._no_fall_through_nodes)
    assert var_14 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_1.Pass(*var_4, **var_5)
    var_7 = [var_6]
    var_8 = []
    var_9 = 'items'
    var_10 = 'body'
    var_11 = {var_9: var_3, var_10: var_7}
    var_12 = module_1.AsyncWith(*var_8, **var_11)
    var_13 = var_2.visit(var_12)
    var_14 = bool(var_12 not in var_2._no_fall_through_nodes)
    assert var_14 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'test'
    var_15 = 'body'
    var_16 = 'orelse'
    var_17 = {var_14: var_7, var_15: var_11, var_16: var_12}
    var_18 = module_1.While(*var_13, **var_17)
    var_19 = var_2.visit(var_18)
    var_20 = bool(var_18 in var_2._no_fall_through_nodes)
    assert var_20 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = 'target'
    var_20 = 'iter'
    var_21 = 'body'
    var_22 = 'orelse'
    var_23 = {var_19: var_7, var_20: var_12, var_21: var_16, var_22: var_17}
    var_24 = module_1.For(*var_18, **var_23)
    var_25 = var_2.visit(var_24)
    var_26 = bool(var_24 not in var_2._no_fall_through_nodes)
    assert var_26 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = 'x'
    var_4 = []
    var_5 = 'id'
    var_6 = {var_5: var_3}
    var_7 = module_1.Name(*var_4, **var_6)
    var_8 = 'y'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.Pass(*var_13, **var_14)
    var_16 = [var_15]
    var_17 = []
    var_18 = []
    var_19 = 'target'
    var_20 = 'iter'
    var_21 = 'body'
    var_22 = 'orelse'
    var_23 = {var_19: var_7, var_20: var_12, var_21: var_16, var_22: var_17}
    var_24 = module_1.AsyncFor(*var_18, **var_23)
    var_25 = var_2.visit(var_24)
    var_26 = bool(var_24 not in var_2._no_fall_through_nodes)
    assert var_26 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = []
    var_9 = {}
    var_10 = module_1.Pass(*var_8, **var_9)
    var_11 = [var_10]
    var_12 = []
    var_13 = []
    var_14 = 'test'
    var_15 = 'body'
    var_16 = 'orelse'
    var_17 = {var_14: var_7, var_15: var_11, var_16: var_12}
    var_18 = module_1.If(*var_13, **var_17)
    var_19 = var_2.visit(var_18)
    var_20 = bool(var_18 not in var_2._no_fall_through_nodes)
    assert var_20 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = True
    var_4 = []
    var_5 = 'value'
    var_6 = {var_5: var_3}
    var_7 = module_1.NameConstant(*var_4, **var_6)
    var_8 = 'x'
    var_9 = []
    var_10 = 'id'
    var_11 = {var_10: var_8}
    var_12 = module_1.Name(*var_9, **var_11)
    var_13 = 'y'
    var_14 = []
    var_15 = 'id'
    var_16 = {var_15: var_13}
    var_17 = module_1.Name(*var_14, **var_16)
    var_18 = []
    var_19 = 'test'
    var_20 = 'body'
    var_21 = 'orelse'
    var_22 = {var_19: var_7, var_20: var_12, var_21: var_17}
    var_23 = module_1.IfExp(*var_18, **var_22)
    var_24 = var_2.visit(var_23)
    var_25 = bool(var_23 not in var_2._no_fall_through_nodes)
    assert var_25 is True

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = []
    var_4 = {}
    var_5 = module_1.Pass(*var_3, **var_4)
    var_6 = [var_5]
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = []
    var_11 = 'body'
    var_12 = 'handlers'
    var_13 = 'orelse'
    var_14 = 'finalbody'
    var_15 = {var_11: var_6, var_12: var_7, var_13: var_8, var_14: var_9}
    var_16 = module_1.Try(*var_10, **var_15)
    var_17 = var_2.visit(var_16)
    var_18 = bool(var_16 not in var_2._no_fall_through_nodes)
    assert var_18 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_init_initializes_no_fall_through_nodes_as_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #4
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #6
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #7
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #8
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #10
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #11
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #15
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #16
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #17
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #20
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #22
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #23
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #24
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #25
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #26
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #27
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #28
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #31
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #33
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #34
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #35
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #36
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #37
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #38
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #39
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #41
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #42
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #43
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #44
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #45
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #46
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #47
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #48
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #49
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #50
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #51
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #52
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_no_fall_through_nodes_initialization. Retrieved 6/7 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = var_2._no_fall_through_nodes
    var_5 = len(var_4)
    assert var_5 == 0



# Parsed testcases at query #54
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #55
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes
    var_4 = len(var_3)
    assert var_4 == 0



# Parsed testcases at query #56
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #57
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_init_creates_no_fall_through_nodes_set. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #59
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #60
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #61
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #62
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #63
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #64
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #65
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #66
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #67
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #68
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #69
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #70
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #71
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #72
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #73
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #74
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #75
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #76
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #77
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_reachability_initialization. Retrieved 9/10 statements.


import vulture.reachability as module_0
import builtins as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = [var_1]
    var_5 = {}
    var_6 = module_1.type(*var_4, **var_5)
    var_7 = isinstance(var_3, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_2._no_fall_through_nodes
    var_10 = var_2._no_fall_through_nodes
    var_11 = len(var_10)
    assert var_11 == 0



# Parsed testcases at query #79
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #80
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #81
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_init_sets_no_fall_through_nodes. Retrieved 4/5 statements.


import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._no_fall_through_nodes



# Parsed testcases at query #83
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #84
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #85
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #86
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #87
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #88
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #89
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #90
#--------------------------




import vulture.reachability as module_0
import builtins as module_1

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = [var_1]
    var_5 = {}
    var_6 = module_1.type(*var_4, **var_5)
    var_7 = isinstance(var_3, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = set()
    var_10 = var_2._no_fall_through_nodes
    var_11 = bool(var_2._no_fall_through_nodes == var_9)
    assert var_11 is True



# Parsed testcases at query #91
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = set()
    var_4 = var_2._no_fall_through_nodes
    var_5 = bool(var_2._no_fall_through_nodes == var_3)
    assert var_5 is True



# Parsed testcases at query #92
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



# Parsed testcases at query #93
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #94
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = bool(not var_2._no_fall_through_nodes)
    assert var_3 is True



# Parsed testcases at query #95
#--------------------------




import vulture.reachability as module_0

def test_case_0():
    var_0 = None
    var_1 = lambda **kwargs: var_0
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2._report
    var_4 = bool(var_2._report == var_1)
    assert var_4 is True
    var_5 = set()
    var_6 = var_2._no_fall_through_nodes
    var_7 = bool(var_2._no_fall_through_nodes == var_5)
    assert var_7 is True



