####################################################################
#        TEST GENERATION BEGINS (CODAMOSA + DeepSeek t=0.8)        #
####################################################################


# Parsed testcases at query #1
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2.reset()
    var_4 = module_1.Pass()
    var_5 = module_1.Pass()
    var_6 = [var_4, var_5]
    var_7 = []
    var_8 = module_1.Module()
    var_9 = var_2.visit(var_8)
    var_10 = len(var_0)
    assert var_10 == 0
    var_11 = var_2.reset()
    var_12 = 'test'
    var_13 = module_1.arguments()
    var_14 = module_1.Break()
    var_15 = module_1.Pass()
    var_16 = [var_14, var_15]
    var_17 = []
    var_18 = module_1.FunctionDef(*var_13)
    var_19 = var_2.visit(var_18)
    var_20 = len(var_0)
    assert var_20 == 1
    var_21 = 'message'
    var_22 = 0
    var_23 = var_0[var_22][var_21]
    var_24 = 'unreachable code after'
    var_25 = var_2.reset()
    var_26 = module_1.arguments()
    var_27 = module_1.Continue()
    var_28 = module_1.Pass()
    var_29 = [var_27, var_28]
    var_30 = []
    var_31 = module_1.AsyncFunctionDef(*var_26)
    var_32 = var_2.visit(var_31)
    var_33 = len(var_0)
    assert var_33 == 1
    var_34 = var_2.reset()
    var_35 = 'open'
    var_36 = module_1.Name()
    var_37 = 'file'
    var_38 = module_1.Constant()
    var_39 = [var_38]
    var_40 = []
    var_41 = module_1.Call(*var_39)
    var_42 = module_1.withitem()
    var_43 = [var_42]
    var_44 = module_1.Pass()
    var_45 = module_1.Pass()
    var_46 = [var_44, var_45]
    var_47 = module_1.With()
    var_48 = var_2.visit(var_47)
    var_49 = len(var_0)
    assert var_49 == 0
    var_50 = var_2.reset()
    var_51 = False
    var_52 = module_1.Constant()
    var_53 = module_1.Pass()
    var_54 = [var_53]
    var_55 = []
    var_56 = module_1.While()
    var_57 = var_2.visit(var_56)
    var_58 = len(var_0)
    assert var_58 == 1
    var_59 = var_2.reset()
    var_60 = True
    var_61 = module_1.Constant()
    var_62 = module_1.Pass()
    var_63 = [var_62]
    var_64 = []
    var_65 = module_1.While()
    var_66 = var_2.visit(var_65)
    var_67 = var_2.reset()
    var_68 = 'i'
    var_69 = module_1.Name()
    var_70 = 'range'
    var_71 = module_1.Name()
    var_72 = 10
    var_73 = module_1.Constant()
    var_74 = [var_73]
    var_75 = []
    var_76 = module_1.Call(*var_74)
    var_77 = module_1.Pass()
    var_78 = [var_77]
    var_79 = []
    var_80 = module_1.For()
    var_81 = var_2.visit(var_80)
    var_82 = len(var_0)
    assert var_82 == 0
    var_83 = var_2.reset()
    var_84 = False
    var_85 = module_1.Constant()
    var_86 = module_1.Pass()
    var_87 = [var_86]
    var_88 = module_1.Pass()
    var_89 = [var_88]
    var_90 = module_1.If()
    var_91 = var_2.visit(var_90)
    var_92 = len(var_0)
    assert var_92 == 1
    var_93 = var_2.reset()
    var_94 = module_1.Constant()
    var_95 = module_1.Pass()
    var_96 = [var_95]
    var_97 = module_1.Pass()
    var_98 = [var_97]
    var_99 = module_1.If()
    var_100 = var_2.visit(var_99)
    var_101 = len(var_0)
    assert var_101 == 1
    var_102 = var_2.reset()
    var_103 = False
    var_104 = module_1.Constant()
    var_105 = module_1.Constant()
    var_106 = 2
    var_107 = module_1.Constant()
    var_108 = module_1.IfExp()
    var_109 = var_2.visit(var_108)
    var_110 = len(var_0)
    assert var_110 == 1
    var_111 = var_2.reset()
    var_112 = module_1.Constant()
    var_113 = module_1.Constant()
    var_114 = module_1.Constant()
    var_115 = module_1.IfExp()
    var_116 = var_2.visit(var_115)
    var_117 = len(var_0)
    assert var_117 == 1
    var_118 = var_2.reset()
    var_119 = module_1.Raise()
    var_120 = [var_119]
    var_121 = 'Exception'
    var_122 = module_1.Name()
    var_123 = None
    var_124 = module_1.Pass()
    var_125 = [var_124]
    var_126 = module_1.ExceptHandler()
    var_127 = [var_126]
    var_128 = module_1.Pass()
    var_129 = [var_128]
    var_130 = []
    var_131 = module_1.Try()
    var_132 = var_2.visit(var_131)
    var_133 = len(var_0)
    assert var_133 == 1
    var_134 = var_2.reset()
    var_135 = module_1.Raise()
    var_136 = [var_135]
    var_137 = module_1.Name()
    var_138 = module_1.Raise()
    var_139 = [var_138]
    var_140 = module_1.ExceptHandler()
    var_141 = [var_140]
    var_142 = []
    var_143 = []
    var_144 = module_1.Try()
    var_145 = var_2.visit(var_144)



# Parsed testcases at query #2
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = var_0.append
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = None
    var_10 = module_1.Return()
    var_11 = var_2.visit(var_10)
    var_12 = var_2.reset()
    var_13 = module_1.Raise()
    var_14 = var_2.visit(var_13)
    var_15 = var_2.reset()
    var_16 = module_1.Pass()
    var_17 = [var_16]
    var_18 = module_1.Module()
    var_19 = var_2.visit(var_18)
    var_20 = len(var_0)
    assert var_20 == 0
    var_21 = var_2.reset()
    var_22 = 'test'
    var_23 = module_1.arguments()
    var_24 = module_1.Pass()
    var_25 = [var_24]
    var_26 = []
    var_27 = module_1.FunctionDef(*var_23)
    var_28 = var_2.visit(var_27)
    var_29 = len(var_0)
    assert var_29 == 0
    var_30 = var_2.reset()
    var_31 = False
    var_32 = module_1.Constant()
    var_33 = module_1.Pass()
    var_34 = [var_33]
    var_35 = []
    var_36 = module_1.While()
    var_37 = var_2.visit(var_36)
    var_38 = len(var_0)
    assert var_38 == 1
    var_39 = var_2.reset()
    var_40 = 'x'
    var_41 = module_1.Name()
    var_42 = 1
    var_43 = module_1.Constant()
    var_44 = [var_43]
    var_45 = module_1.List()
    var_46 = module_1.Pass()
    var_47 = [var_46]
    var_48 = []
    var_49 = module_1.For()
    var_50 = var_2.visit(var_49)
    var_51 = len(var_0)
    assert var_51 == 0
    var_52 = var_2.reset()
    var_53 = True
    var_54 = module_1.Constant()
    var_55 = module_1.Pass()
    var_56 = [var_55]
    var_57 = module_1.Pass()
    var_58 = [var_57]
    var_59 = module_1.If()
    var_60 = var_2.visit(var_59)
    var_61 = len(var_0)
    var_62 = var_2.reset()
    var_63 = module_1.Constant()
    var_64 = module_1.Constant()
    var_65 = 2
    var_66 = module_1.Constant()
    var_67 = module_1.IfExp()
    var_68 = var_2.visit(var_67)
    var_69 = len(var_0)
    assert var_69 == 1
    var_70 = var_2.reset()
    var_71 = module_1.Pass()
    var_72 = [var_71]
    var_73 = module_1.Pass()
    var_74 = [var_73]
    var_75 = module_1.ExceptHandler()
    var_76 = [var_75]
    var_77 = []
    var_78 = []
    var_79 = module_1.Try()
    var_80 = var_2.visit(var_79)
    var_81 = len(var_0)
    assert var_81 == 0
    var_82 = var_2.reset()
    var_83 = module_1.Break()
    var_84 = module_1.Pass()
    var_85 = [var_83, var_84]
    var_86 = module_1.Module()
    var_87 = var_2.visit(var_86)
    var_88 = len(var_0)



# Parsed testcases at query #3
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: reports.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = None
    var_10 = module_1.Constant()
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = 1
    var_18 = module_1.Constant()
    var_19 = module_1.Return()
    var_20 = 2
    var_21 = module_1.Constant()
    var_22 = module_1.Expr()
    var_23 = [var_19, var_22]
    var_24 = []
    var_25 = module_1.Module()
    var_26 = len(var_0)
    assert var_26 == 1
    var_27 = var_2.reset()
    var_28 = False
    var_29 = module_1.Constant()
    var_30 = module_1.Pass()
    var_31 = [var_30]
    var_32 = []
    var_33 = module_1.If()
    var_34 = var_2.visit(var_33)
    var_35 = len(var_0)
    assert var_35 == 1
    var_36 = var_2.reset()
    var_37 = True
    var_38 = module_1.Constant()
    var_39 = module_1.Pass()
    var_40 = [var_39]
    var_41 = module_1.Pass()
    var_42 = [var_41]
    var_43 = module_1.If()
    var_44 = var_2.visit(var_43)
    var_45 = len(var_0)
    assert var_45 == 1
    var_46 = var_2.reset()
    var_47 = True
    var_48 = module_1.Constant()
    var_49 = module_1.Pass()
    var_50 = [var_49]
    var_51 = []
    var_52 = module_1.If()
    var_53 = var_2.visit(var_52)
    var_54 = len(var_0)
    assert var_54 == 1
    var_55 = var_2.reset()
    var_56 = module_1.Constant()
    var_57 = module_1.Pass()
    var_58 = [var_57]
    var_59 = []
    var_60 = module_1.While()
    var_61 = var_2.visit(var_60)
    var_62 = len(var_0)
    assert var_62 == 1
    var_63 = var_2.reset()
    var_64 = True
    var_65 = module_1.Constant()
    var_66 = module_1.Break()
    var_67 = [var_66]
    var_68 = []
    var_69 = module_1.While()
    var_70 = var_2.visit(var_69)
    var_71 = len(var_0)
    assert var_71 == 0
    var_72 = var_2.reset()
    var_73 = True
    var_74 = module_1.Constant()
    var_75 = module_1.Pass()
    var_76 = [var_75]
    var_77 = []
    var_78 = module_1.While()
    var_79 = var_2.visit(var_78)
    var_80 = var_2.reset()
    var_81 = module_1.Constant()
    var_82 = module_1.Return()
    var_83 = [var_82]
    var_84 = module_1.Pass()
    var_85 = [var_84]
    var_86 = module_1.ExceptHandler()
    var_87 = [var_86]
    var_88 = module_1.Pass()
    var_89 = [var_88]
    var_90 = []
    var_91 = module_1.Try()
    var_92 = var_2.visit(var_91)
    var_93 = len(var_0)
    assert var_93 == 1
    var_94 = var_2.reset()
    var_95 = module_1.Constant()
    var_96 = module_1.Constant()
    var_97 = module_1.Constant()
    var_98 = module_1.IfExp()
    var_99 = var_2.visit(var_98)
    var_100 = len(var_0)
    assert var_100 == 1
    var_101 = var_2.reset()
    var_102 = True
    var_103 = module_1.Constant()
    var_104 = module_1.Constant()
    var_105 = module_1.Constant()
    var_106 = module_1.IfExp()
    var_107 = var_2.visit(var_106)
    var_108 = len(var_0)
    assert var_108 == 1
    var_109 = var_2.reset()
    var_110 = module_1.Pass()
    var_111 = var_2.visit(var_110)



# Parsed testcases at query #4
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = 1
    var_10 = module_1.Constant()
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = False
    var_18 = module_1.Pass()
    var_19 = [var_18]
    var_20 = 'message'
    var_21 = "unsatisfiable 'if' condition"
    var_22 = var_2.reset()
    var_23 = True
    var_24 = module_1.Pass()
    var_25 = [var_24]
    var_26 = module_1.Pass()
    var_27 = [var_26]
    var_28 = "unreachable 'else' block"
    var_29 = var_2.reset()
    var_30 = True
    var_31 = module_1.Pass()
    var_32 = [var_31]
    var_33 = 'redundant if-condition'
    var_34 = var_2.reset()
    var_35 = module_1.Constant()
    var_36 = module_1.Pass()
    var_37 = [var_36]
    var_38 = []
    var_39 = module_1.While()
    var_40 = var_2.visit(var_39)
    var_41 = "unsatisfiable 'while' condition"
    var_42 = var_2.reset()
    var_43 = True
    var_44 = module_1.Constant()
    var_45 = module_1.Pass()
    var_46 = [var_45]
    var_47 = []
    var_48 = module_1.While()
    var_49 = var_2.visit(var_48)
    var_50 = var_2.reset()
    var_51 = module_1.Constant()
    var_52 = module_1.Return()
    var_53 = [var_52]
    var_54 = module_1.Pass()
    var_55 = [var_54]
    var_56 = None
    var_57 = module_1.ExceptHandler()
    var_58 = [var_57]
    var_59 = []
    var_60 = []
    var_61 = module_1.Try()
    var_62 = var_2.visit(var_61)
    var_63 = var_2.reset()
    var_64 = module_1.Constant()
    var_65 = module_1.Return()
    var_66 = [var_65]
    var_67 = module_1.Pass()
    var_68 = [var_67]
    var_69 = module_1.ExceptHandler()
    var_70 = [var_69]
    var_71 = module_1.Pass()
    var_72 = [var_71]
    var_73 = []
    var_74 = module_1.Try()
    var_75 = var_2.visit(var_74)
    var_76 = var_2.reset()
    var_77 = module_1.Pass()
    var_78 = var_2.visit(var_77)



# Parsed testcases at query #5
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = 1
    var_10 = module_1.Constant()
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = module_1.Pass()
    var_18 = module_1.Break()
    var_19 = module_1.Pass()
    var_20 = [var_17, var_18, var_19]
    var_21 = module_1.Module()
    var_22 = var_2.visit(var_21)
    var_23 = var_21.body[var_9]
    var_24 = var_2.reset()
    var_25 = 'test_func'
    var_26 = module_1.arguments()
    var_27 = module_1.Constant()
    var_28 = module_1.Return()
    var_29 = module_1.Pass()
    var_30 = [var_28, var_29]
    var_31 = []
    var_32 = module_1.FunctionDef(*var_26)
    var_33 = var_2.visit(var_32)
    var_34 = 0
    var_35 = var_32.body[var_34]
    var_36 = len(var_0)
    assert var_36 == 1
    var_37 = 'message'
    var_38 = var_0[var_34][var_37]
    var_39 = var_2.reset()
    var_40 = False
    var_41 = module_1.Constant()
    var_42 = module_1.Pass()
    var_43 = [var_42]
    var_44 = []
    var_45 = module_1.While()
    var_46 = var_2.visit(var_45)
    var_47 = len(var_0)
    assert var_47 == 1
    var_48 = var_0[var_40][var_37]
    var_49 = var_2.reset()
    var_50 = True
    var_51 = module_1.Constant()
    var_52 = module_1.Break()
    var_53 = [var_52]
    var_54 = []
    var_55 = module_1.While()
    var_56 = var_2.visit(var_55)
    var_57 = var_2.reset()
    var_58 = False
    var_59 = module_1.Constant()
    var_60 = module_1.Pass()
    var_61 = [var_60]
    var_62 = []
    var_63 = module_1.If()
    var_64 = var_2.visit(var_63)
    var_65 = len(var_0)
    assert var_65 == 1
    var_66 = var_0[var_58][var_37]
    var_67 = var_2.reset()
    var_68 = True
    var_69 = module_1.Constant()
    var_70 = module_1.Pass()
    var_71 = [var_70]
    var_72 = module_1.Pass()
    var_73 = [var_72]
    var_74 = module_1.If()
    var_75 = var_2.visit(var_74)
    var_76 = len(var_0)
    var_77 = 'unreachable'
    var_78 = [r for r in var_0 if var_77 in r[var_40].lower()]
    var_79 = len(var_78)
    var_80 = var_2.reset()
    var_81 = False
    var_82 = module_1.Constant()
    var_83 = module_1.Constant()
    var_84 = 2
    var_85 = module_1.Constant()
    var_86 = module_1.IfExp()
    var_87 = var_2.visit(var_86)
    var_88 = len(var_0)
    assert var_88 == 1
    var_89 = var_0[var_81][var_37]
    var_90 = var_2.reset()
    var_91 = True
    var_92 = module_1.Constant()
    var_93 = module_1.Constant()
    var_94 = module_1.Constant()
    var_95 = module_1.IfExp()
    var_96 = var_2.visit(var_95)
    var_97 = len(var_0)
    assert var_97 == 1
    var_98 = var_0[var_81][var_37]
    var_99 = var_2.reset()
    var_100 = module_1.Raise()
    var_101 = [var_100]
    var_102 = module_1.Raise()
    var_103 = [var_102]
    var_104 = 'Exception'
    var_105 = module_1.Name()
    var_106 = None
    var_107 = module_1.ExceptHandler()
    var_108 = [var_107]
    var_109 = []
    var_110 = []
    var_111 = module_1.Try()
    var_112 = var_2.visit(var_111)



# Parsed testcases at query #6
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = module_0.Return()
    var_4 = module_0.Break()
    var_5 = module_0.Continue()
    var_6 = module_0.Raise()
    var_7 = module_0.Pass()
    var_8 = module_0.Pass()
    var_9 = [var_7, var_8]
    var_10 = module_0.Module()
    var_11 = module_0.Pass()
    var_12 = module_0.Pass()
    var_13 = [var_11, var_12]
    var_14 = 'test_func'
    var_15 = []
    var_16 = module_0.arguments()
    var_17 = module_0.FunctionDef(*var_16)
    var_18 = False
    var_19 = module_0.Constant()
    var_20 = module_0.Pass()
    var_21 = [var_20]
    var_22 = []
    var_23 = module_0.While()
    var_24 = len(var_0)
    var_25 = 'x'
    var_26 = module_0.Load()
    var_27 = module_0.Name()
    var_28 = module_0.Pass()
    var_29 = [var_28]
    var_30 = []
    var_31 = module_0.If()
    var_32 = len(var_0)
    var_33 = len(var_0)
    var_34 = True
    var_35 = module_0.Constant()
    var_36 = module_0.Constant()
    var_37 = 2
    var_38 = module_0.Constant()
    var_39 = module_0.IfExp()
    var_40 = len(var_0)
    var_41 = module_0.Pass()
    var_42 = [var_41]
    var_43 = []
    var_44 = []
    var_45 = []
    var_46 = module_0.Try()
    var_47 = len(var_0)
    var_48 = len(var_0)



# Parsed testcases at query #7
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = module_1.Return()
    var_10 = var_2.visit(var_9)
    var_11 = var_2.reset()
    var_12 = module_1.Raise()
    var_13 = var_2.visit(var_12)
    var_14 = var_2.reset()
    var_15 = module_1.Pass()
    var_16 = [var_15]
    var_17 = module_1.Module()
    var_18 = var_2.visit(var_17)
    var_19 = var_2.reset()
    var_20 = 'test'
    var_21 = module_1.Pass()
    var_22 = [var_21]
    var_23 = module_1.arguments()
    var_24 = []
    var_25 = module_1.FunctionDef(*var_23)
    var_26 = var_2.visit(var_25)
    var_27 = var_2.reset()
    var_28 = True
    var_29 = module_1.Constant()
    var_30 = module_1.Pass()
    var_31 = [var_30]
    var_32 = []
    var_33 = module_1.While()
    var_34 = var_2.visit(var_33)
    var_35 = var_2.reset()
    var_36 = False
    var_37 = module_1.Constant()
    var_38 = module_1.Pass()
    var_39 = [var_38]
    var_40 = []
    var_41 = module_1.While()
    var_42 = var_2.visit(var_41)
    var_43 = len(var_0)
    assert var_43 == 1
    var_44 = var_2.reset()
    var_45 = 'x'
    var_46 = module_1.Store()
    var_47 = module_1.Name()
    var_48 = []
    var_49 = module_1.Load()
    var_50 = module_1.List()
    var_51 = module_1.Pass()
    var_52 = [var_51]
    var_53 = []
    var_54 = module_1.For()
    var_55 = var_2.visit(var_54)
    var_56 = var_2.reset()
    var_57 = module_1.Constant()
    var_58 = module_1.Pass()
    var_59 = [var_58]
    var_60 = module_1.Pass()
    var_61 = [var_60]
    var_62 = module_1.If()
    var_63 = var_2.visit(var_62)
    var_64 = len(var_0)
    assert var_64 == 1
    var_65 = var_2.reset()
    var_66 = module_1.Constant()
    var_67 = module_1.Pass()
    var_68 = [var_67]
    var_69 = module_1.Pass()
    var_70 = [var_69]
    var_71 = module_1.If()
    var_72 = var_2.visit(var_71)
    var_73 = len(var_0)
    assert var_73 == 1
    var_74 = var_2.reset()
    var_75 = module_1.Constant()
    var_76 = module_1.Constant()
    var_77 = 2
    var_78 = module_1.Constant()
    var_79 = module_1.IfExp()
    var_80 = var_2.visit(var_79)
    var_81 = len(var_0)
    assert var_81 == 1
    var_82 = var_2.reset()
    var_83 = module_1.Raise()
    var_84 = [var_83]
    var_85 = module_1.Pass()
    var_86 = [var_85]
    var_87 = module_1.ExceptHandler()
    var_88 = [var_87]
    var_89 = module_1.Pass()
    var_90 = [var_89]
    var_91 = []
    var_92 = module_1.Try()
    var_93 = var_2.visit(var_92)
    var_94 = len(var_0)
    assert var_94 == 1



# Parsed testcases at query #8
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = len(var_0)
    assert var_3 == 0
    var_4 = []
    var_5 = lambda **kwargs: report_calls.append(kwargs)
    var_6 = module_0.Reachability(var_5)
    var_7 = module_1.Pass()
    var_8 = module_1.Pass()
    var_9 = [var_7, var_8]
    var_10 = []
    var_11 = module_1.Module()
    var_12 = var_6.visit(var_11)
    var_13 = len(var_4)
    assert var_13 == 0
    var_14 = []
    var_15 = lambda **kwargs: report_calls.append(kwargs)
    var_16 = module_0.Reachability(var_15)
    var_17 = 'test'
    var_18 = []
    var_19 = []
    var_20 = []
    var_21 = []
    var_22 = []
    var_23 = module_1.arguments(*var_19)
    var_24 = module_1.Pass()
    var_25 = module_1.Pass()
    var_26 = [var_24, var_25]
    var_27 = []
    var_28 = module_1.FunctionDef(*var_23)
    var_29 = var_16.visit(var_28)
    var_30 = len(var_14)
    assert var_30 == 0
    var_31 = []
    var_32 = lambda **kwargs: report_calls.append(kwargs)
    var_33 = module_0.Reachability(var_32)
    var_34 = False
    var_35 = module_1.Constant()
    var_36 = module_1.Pass()
    var_37 = [var_36]
    var_38 = []
    var_39 = module_1.While()
    var_40 = var_33.visit(var_39)
    var_41 = len(var_31)
    assert var_41 == 1
    var_42 = []
    var_43 = lambda **kwargs: report_calls.append(kwargs)
    var_44 = module_0.Reachability(var_43)
    var_45 = True
    var_46 = module_1.Constant()
    var_47 = module_1.Break()
    var_48 = [var_47]
    var_49 = []
    var_50 = module_1.While()
    var_51 = var_44.visit(var_50)
    var_52 = len(var_42)
    assert var_52 == 0
    var_53 = []
    var_54 = lambda **kwargs: report_calls.append(kwargs)
    var_55 = module_0.Reachability(var_54)
    var_56 = module_1.Constant()
    var_57 = module_1.Pass()
    var_58 = [var_57]
    var_59 = []
    var_60 = module_1.While()
    var_61 = var_55.visit(var_60)
    var_62 = []
    var_63 = lambda **kwargs: report_calls.append(kwargs)
    var_64 = module_0.Reachability(var_63)
    var_65 = 'i'
    var_66 = module_1.Store()
    var_67 = module_1.Name()
    var_68 = 'range'
    var_69 = module_1.Load()
    var_70 = module_1.Name()
    var_71 = 10
    var_72 = module_1.Constant()
    var_73 = [var_72]
    var_74 = []
    var_75 = module_1.Call(*var_73)
    var_76 = module_1.Pass()
    var_77 = module_1.Pass()
    var_78 = [var_76, var_77]
    var_79 = []
    var_80 = module_1.For()
    var_81 = var_64.visit(var_80)
    var_82 = len(var_62)
    assert var_82 == 0
    var_83 = []
    var_84 = lambda **kwargs: report_calls.append(kwargs)
    var_85 = module_0.Reachability(var_84)
    var_86 = module_1.Constant()
    var_87 = module_1.Pass()
    var_88 = [var_87]
    var_89 = module_1.Pass()
    var_90 = [var_89]
    var_91 = module_1.If()
    var_92 = var_85.visit(var_91)
    var_93 = len(var_83)
    assert var_93 == 1
    var_94 = []
    var_95 = lambda **kwargs: report_calls.append(kwargs)
    var_96 = module_0.Reachability(var_95)
    var_97 = module_1.Constant()
    var_98 = module_1.Pass()
    var_99 = [var_98]
    var_100 = module_1.Pass()
    var_101 = [var_100]
    var_102 = module_1.If()
    var_103 = var_96.visit(var_102)
    var_104 = len(var_94)
    assert var_104 == 1
    var_105 = []
    var_106 = lambda **kwargs: report_calls.append(kwargs)
    var_107 = module_0.Reachability(var_106)
    var_108 = module_1.Constant()
    var_109 = module_1.Pass()
    var_110 = [var_109]
    var_111 = []
    var_112 = module_1.If()
    var_113 = var_107.visit(var_112)
    var_114 = len(var_105)
    assert var_114 == 1
    var_115 = []
    var_116 = lambda **kwargs: report_calls.append(kwargs)
    var_117 = module_0.Reachability(var_116)
    var_118 = module_1.Constant()
    var_119 = module_1.Constant()
    var_120 = 2
    var_121 = module_1.Constant()
    var_122 = module_1.IfExp()
    var_123 = var_117.visit(var_122)
    var_124 = len(var_115)
    assert var_124 == 1
    var_125 = []
    var_126 = lambda **kwargs: report_calls.append(kwargs)
    var_127 = module_0.Reachability(var_126)
    var_128 = module_1.Constant()
    var_129 = module_1.Constant()
    var_130 = module_1.Constant()
    var_131 = module_1.IfExp()
    var_132 = var_127.visit(var_131)
    var_133 = len(var_125)
    assert var_133 == 1
    var_134 = []
    var_135 = lambda **kwargs: report_calls.append(kwargs)
    var_136 = module_0.Reachability(var_135)
    var_137 = module_1.Raise()
    var_138 = [var_137]
    var_139 = 'Exception'
    var_140 = module_1.Load()
    var_141 = module_1.Name()
    var_142 = None
    var_143 = module_1.Pass()
    var_144 = [var_143]
    var_145 = module_1.ExceptHandler()
    var_146 = [var_145]
    var_147 = module_1.Pass()
    var_148 = [var_147]
    var_149 = []
    var_150 = module_1.Try()
    var_151 = var_136.visit(var_150)
    var_152 = len(var_134)
    assert var_152 == 1
    var_153 = []
    var_154 = lambda **kwargs: report_calls.append(kwargs)
    var_155 = module_0.Reachability(var_154)
    var_156 = module_1.Raise()
    var_157 = [var_156]
    var_158 = module_1.Load()
    var_159 = module_1.Name()
    var_160 = module_1.Raise()
    var_161 = [var_160]
    var_162 = module_1.ExceptHandler()
    var_163 = [var_162]
    var_164 = []
    var_165 = []
    var_166 = module_1.Try()
    var_167 = var_155.visit(var_166)
    var_168 = []
    var_169 = lambda **kwargs: report_calls.append(kwargs)
    var_170 = module_0.Reachability(var_169)
    var_171 = module_1.Constant()
    var_172 = module_1.Return()
    var_173 = var_170.visit(var_172)
    var_174 = var_170.reset()



# Parsed testcases at query #9
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = module_0.Constant()
    var_3 = module_0.Return()
    var_4 = 2
    var_5 = module_0.Constant()
    var_6 = module_0.Expr()
    var_7 = [var_3, var_6]
    var_8 = []
    var_9 = module_0.Module()
    var_10 = len(var_0)
    var_11 = str(var_0)
    var_12 = False
    var_13 = module_0.Constant()
    var_14 = module_0.Pass()
    var_15 = [var_14]
    var_16 = []
    var_17 = module_0.While()
    var_18 = True
    var_19 = len(var_0)
    var_20 = str(var_0)
    var_21 = True
    var_22 = module_0.Constant()
    var_23 = module_0.Pass()
    var_24 = [var_23]
    var_25 = module_0.Pass()
    var_26 = [var_25]
    var_27 = module_0.If()
    var_28 = True
    var_29 = len(var_0)
    var_30 = "unreachable 'else' block"
    var_31 = module_0.Constant()
    var_32 = module_0.Constant()
    var_33 = module_0.Constant()
    var_34 = module_0.IfExp()
    var_35 = True
    var_36 = len(var_0)
    var_37 = str(var_0)
    var_38 = module_0.Constant()
    var_39 = module_0.Return()
    var_40 = [var_39]
    var_41 = []
    var_42 = module_0.Pass()
    var_43 = [var_42]
    var_44 = []
    var_45 = module_0.Try()
    var_46 = len(var_0)
    var_47 = str(var_0)
    var_48 = module_0.Pass()
    var_49 = len(var_0)
    assert var_49 == 0
    var_50 = 'x'
    var_51 = module_0.Name()
    var_52 = module_0.Constant()
    var_53 = [var_52]
    var_54 = module_0.List()
    var_55 = module_0.Pass()
    var_56 = [var_55]
    var_57 = []
    var_58 = module_0.For()
    var_59 = len(var_0)
    assert var_59 == 0
    var_60 = 'ctx'
    var_61 = module_0.Name()
    var_62 = module_0.withitem()
    var_63 = [var_62]
    var_64 = module_0.Pass()
    var_65 = [var_64]
    var_66 = module_0.With()
    var_67 = len(var_0)
    assert var_67 == 0



# Parsed testcases at query #10
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Break node is marked as no fall through.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Break()
    var_5 = var_3.visit(var_4)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Continue node is marked as no fall through.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Continue()
    var_5 = var_3.visit(var_4)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Return node is marked as no fall through.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = 1
    var_5 = module_1.Constant()
    var_6 = module_1.Return()
    var_7 = var_3.visit(var_6)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Raise node is marked as no fall through.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Raise()
    var_5 = var_3.visit(var_4)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Module node analyzes its body statements.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Break()
    var_5 = 1
    var_6 = module_1.Constant()
    var_7 = module_1.Expr()
    var_8 = [var_4, var_7]
    var_9 = []
    var_10 = module_1.Module()
    var_11 = var_3.visit(var_10)
    var_12 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that FunctionDef node analyzes its body.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = 'test_func'
    var_5 = []
    var_6 = []
    var_7 = []
    var_8 = []
    var_9 = []
    var_10 = module_1.arguments(*var_6)
    var_11 = module_1.Break()
    var_12 = 1
    var_13 = module_1.Constant()
    var_14 = module_1.Expr()
    var_15 = [var_11, var_14]
    var_16 = []
    var_17 = None
    var_18 = module_1.FunctionDef(*var_10)
    var_19 = var_3.visit(var_18)
    var_20 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that While node is handled correctly.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = True
    var_5 = module_1.Constant()
    var_6 = module_1.Break()
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.While()
    var_10 = var_3.visit(var_9)
    var_11 = len(var_1)
    assert var_11 == 0

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that For node analyzes its body.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = 'x'
    var_5 = module_1.Store()
    var_6 = module_1.Name()
    var_7 = 1
    var_8 = module_1.Constant()
    var_9 = [var_8]
    var_10 = module_1.Load()
    var_11 = module_1.List()
    var_12 = module_1.Break()
    var_13 = module_1.Constant()
    var_14 = module_1.Expr()
    var_15 = [var_12, var_14]
    var_16 = []
    var_17 = module_1.For()
    var_18 = var_3.visit(var_17)
    var_19 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that If node with always false condition is reported.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = False
    var_5 = module_1.Constant()
    var_6 = module_1.Pass()
    var_7 = [var_6]
    var_8 = []
    var_9 = module_1.If()
    var_10 = var_3.visit(var_9)
    var_11 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that If node with always true condition is reported.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = True
    var_5 = module_1.Constant()
    var_6 = module_1.Pass()
    var_7 = [var_6]
    var_8 = module_1.Pass()
    var_9 = [var_8]
    var_10 = module_1.If()
    var_11 = var_3.visit(var_10)
    var_12 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that IfExp node with always false condition is reported.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = False
    var_5 = module_1.Constant()
    var_6 = 1
    var_7 = module_1.Constant()
    var_8 = 2
    var_9 = module_1.Constant()
    var_10 = module_1.IfExp()
    var_11 = var_3.visit(var_10)
    var_12 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that IfExp node with always true condition is reported.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = True
    var_5 = module_1.Constant()
    var_6 = module_1.Constant()
    var_7 = 2
    var_8 = module_1.Constant()
    var_9 = module_1.IfExp()
    var_10 = var_3.visit(var_9)
    var_11 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Try node analyzes its body.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Raise()
    var_5 = [var_4]
    var_6 = 'Exception'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = None
    var_10 = module_1.Pass()
    var_11 = [var_10]
    var_12 = module_1.ExceptHandler()
    var_13 = [var_12]
    var_14 = module_1.Pass()
    var_15 = [var_14]
    var_16 = []
    var_17 = module_1.Try()
    var_18 = var_3.visit(var_17)
    var_19 = len(var_1)

import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = 'Test that Try node analyzes its body.'
    var_1 = []
    var_2 = var_1.append
    var_3 = module_0.Reachability(var_2)
    var_4 = module_1.Raise()
    var_5 = [var_4]
    var_6 = 'Exception'
    var_7 = module_1.Load()
    var_8 = module_1.Name()
    var_9 = None
    var_10 = module_1.Pass()
    var_11 = [var_10]
    var_12 = module_1.ExceptHandler()
    var_13 = [var_12]
    var_14 = module_1.Pass()
    var_15 = [var_14]
    var_16 = []
    var_17 = module_1.Try()
    var_18 = var_3.visit(var_17)
    var_19 = len(var_1)



# Parsed testcases at query #11
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Continue()
    var_3 = None
    var_4 = module_0.Constant()
    var_5 = module_0.Return()
    var_6 = module_0.Raise()
    var_7 = module_0.Constant()
    var_8 = module_0.Return()
    var_9 = 1
    var_10 = module_0.Constant()
    var_11 = module_0.Expr()
    var_12 = [var_8, var_11]
    var_13 = []
    var_14 = module_0.Module()
    var_15 = len(var_0)
    var_16 = "unreachable code after 'return'"
    var_17 = module_0.Constant()
    var_18 = module_0.Lt()
    var_19 = [var_18]
    var_20 = 0
    var_21 = module_0.Constant()
    var_22 = [var_21]
    var_23 = module_0.Compare()
    var_24 = module_0.Constant()
    var_25 = module_0.Expr()
    var_26 = [var_25]
    var_27 = []
    var_28 = module_0.If()
    var_29 = "unsatisfiable 'if' condition"
    var_30 = module_0.Constant()
    var_31 = module_0.Gt()
    var_32 = [var_31]
    var_33 = module_0.Constant()
    var_34 = [var_33]
    var_35 = module_0.Compare()
    var_36 = module_0.Constant()
    var_37 = module_0.Expr()
    var_38 = [var_37]
    var_39 = 2
    var_40 = module_0.Constant()
    var_41 = module_0.Expr()
    var_42 = [var_41]
    var_43 = module_0.If()
    var_44 = "unreachable 'else' block"
    var_45 = module_0.Constant()
    var_46 = module_0.Lt()
    var_47 = [var_46]
    var_48 = module_0.Constant()
    var_49 = [var_48]
    var_50 = module_0.Compare()
    var_51 = module_0.Constant()
    var_52 = module_0.Expr()
    var_53 = [var_52]
    var_54 = []
    var_55 = module_0.While()
    var_56 = "unsatisfiable 'while' condition"
    var_57 = module_0.Constant()
    var_58 = module_0.Return()
    var_59 = [var_58]
    var_60 = module_0.Constant()
    var_61 = module_0.Expr()
    var_62 = [var_61]
    var_63 = module_0.ExceptHandler()
    var_64 = [var_63]
    var_65 = module_0.Constant()
    var_66 = module_0.Expr()
    var_67 = [var_66]
    var_68 = []
    var_69 = module_0.Try()
    var_70 = module_0.Constant()
    var_71 = module_0.Lt()
    var_72 = [var_71]
    var_73 = module_0.Constant()
    var_74 = [var_73]
    var_75 = module_0.Compare()
    var_76 = module_0.Constant()
    var_77 = module_0.Constant()
    var_78 = module_0.IfExp()
    var_79 = "unsatisfiable 'ternary' condition"



# Parsed testcases at query #12
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Return()



# Parsed testcases at query #13
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Continue()



# Parsed testcases at query #14
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Return()



# Parsed testcases at query #15
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Continue()
    var_3 = None
    var_4 = module_0.Constant()
    var_5 = module_0.Return()
    var_6 = module_0.Raise()
    var_7 = 1
    var_8 = module_0.Constant()
    var_9 = module_0.Return()
    var_10 = 2
    var_11 = module_0.Constant()
    var_12 = module_0.Expr()
    var_13 = [var_9, var_12]
    var_14 = []
    var_15 = module_0.Module()
    var_16 = len(var_0)
    assert var_16 == 1
    var_17 = False
    var_18 = module_0.Constant()
    var_19 = module_0.Constant()
    var_20 = module_0.Expr()
    var_21 = [var_20]
    var_22 = []
    var_23 = module_0.If()
    var_24 = len(var_0)
    assert var_24 == 1
    var_25 = True
    var_26 = module_0.Constant()
    var_27 = module_0.Constant()
    var_28 = module_0.Expr()
    var_29 = [var_28]
    var_30 = []
    var_31 = module_0.If()
    var_32 = len(var_0)
    assert var_32 == 1
    var_33 = module_0.Constant()
    var_34 = module_0.Constant()
    var_35 = module_0.Expr()
    var_36 = [var_35]
    var_37 = []
    var_38 = module_0.While()
    var_39 = len(var_0)
    assert var_39 == 1
    var_40 = True
    var_41 = module_0.Constant()
    var_42 = module_0.Constant()
    var_43 = module_0.Expr()
    var_44 = [var_43]
    var_45 = []
    var_46 = module_0.While()
    var_47 = True
    var_48 = module_0.Constant()
    var_49 = module_0.Break()
    var_50 = [var_49]
    var_51 = []
    var_52 = module_0.While()
    var_53 = module_0.Constant()
    var_54 = module_0.Constant()
    var_55 = module_0.Constant()
    var_56 = module_0.IfExp()
    var_57 = len(var_0)
    assert var_57 == 1
    var_58 = True
    var_59 = module_0.Constant()
    var_60 = module_0.Constant()
    var_61 = module_0.Constant()
    var_62 = module_0.IfExp()
    var_63 = len(var_0)
    assert var_63 == 1
    var_64 = module_0.Constant()
    var_65 = module_0.Return()
    var_66 = [var_65]
    var_67 = 'Exception'
    var_68 = module_0.Load()
    var_69 = module_0.Name()
    var_70 = module_0.Constant()
    var_71 = module_0.Expr()
    var_72 = [var_71]
    var_73 = module_0.ExceptHandler()
    var_74 = [var_73]
    var_75 = 3
    var_76 = module_0.Constant()
    var_77 = module_0.Expr()
    var_78 = [var_77]
    var_79 = []
    var_80 = module_0.Try()
    var_81 = len(var_0)
    assert var_81 == 1
    var_82 = 'x'
    var_83 = module_0.Store()
    var_84 = module_0.Name()
    var_85 = module_0.Constant()
    var_86 = [var_85]
    var_87 = module_0.Load()
    var_88 = module_0.List()
    var_89 = module_0.Constant()
    var_90 = module_0.Expr()
    var_91 = [var_90]
    var_92 = []
    var_93 = module_0.For()
    var_94 = len(var_0)
    assert var_94 == 0
    var_95 = 'test_func'
    var_96 = []
    var_97 = []
    var_98 = []
    var_99 = []
    var_100 = []
    var_101 = module_0.arguments(*var_97)
    var_102 = module_0.Constant()
    var_103 = module_0.Return()
    var_104 = module_0.Constant()
    var_105 = module_0.Expr()
    var_106 = [var_103, var_105]
    var_107 = []
    var_108 = module_0.FunctionDef(*var_101)
    var_109 = len(var_0)
    assert var_109 == 1



# Parsed testcases at query #16
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = 'test_node'
    var_3 = set()



# Parsed testcases at query #17
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = len(var_0)
    assert var_5 == 0
    var_6 = []
    var_7 = lambda **kwargs: report_calls.append(kwargs)
    var_8 = module_0.Reachability(var_7)
    var_9 = module_1.Continue()
    var_10 = var_8.visit(var_9)
    var_11 = len(var_6)
    assert var_11 == 0
    var_12 = []
    var_13 = lambda **kwargs: report_calls.append(kwargs)
    var_14 = module_0.Reachability(var_13)
    var_15 = None
    var_16 = module_1.Return()
    var_17 = var_14.visit(var_16)
    var_18 = len(var_12)
    assert var_18 == 0
    var_19 = []
    var_20 = lambda **kwargs: report_calls.append(kwargs)
    var_21 = module_0.Reachability(var_20)
    var_22 = module_1.Raise()
    var_23 = var_21.visit(var_22)
    var_24 = len(var_19)
    assert var_24 == 0
    var_25 = []
    var_26 = lambda **kwargs: report_calls.append(kwargs)
    var_27 = module_0.Reachability(var_26)
    var_28 = []
    var_29 = []
    var_30 = module_1.Module()
    var_31 = var_27.visit(var_30)
    var_32 = len(var_25)
    assert var_32 == 0
    var_33 = []
    var_34 = lambda **kwargs: report_calls.append(kwargs)
    var_35 = module_0.Reachability(var_34)
    var_36 = module_1.Pass()
    var_37 = [var_36]
    var_38 = []
    var_39 = module_1.Module()
    var_40 = var_35.visit(var_39)
    var_41 = len(var_33)
    assert var_41 == 0
    var_42 = []
    var_43 = lambda **kwargs: report_calls.append(kwargs)
    var_44 = module_0.Reachability(var_43)
    var_45 = 'test'
    var_46 = []
    var_47 = []
    var_48 = []
    var_49 = []
    var_50 = []
    var_51 = module_1.arguments(*var_47)
    var_52 = module_1.Pass()
    var_53 = [var_52]
    var_54 = []
    var_55 = module_1.FunctionDef(*var_51)
    var_56 = var_44.visit(var_55)
    var_57 = len(var_42)
    assert var_57 == 0
    var_58 = []
    var_59 = lambda **kwargs: report_calls.append(kwargs)
    var_60 = module_0.Reachability(var_59)
    var_61 = []
    var_62 = []
    var_63 = []
    var_64 = []
    var_65 = []
    var_66 = module_1.arguments(*var_62)
    var_67 = module_1.Pass()
    var_68 = [var_67]
    var_69 = []
    var_70 = module_1.AsyncFunctionDef(*var_66)
    var_71 = var_60.visit(var_70)
    var_72 = len(var_58)
    assert var_72 == 0
    var_73 = []
    var_74 = lambda **kwargs: report_calls.append(kwargs)
    var_75 = module_0.Reachability(var_74)
    var_76 = 'x'
    var_77 = module_1.Load()
    var_78 = module_1.Name()
    var_79 = module_1.withitem()
    var_80 = [var_79]
    var_81 = module_1.Pass()
    var_82 = [var_81]
    var_83 = module_1.With()
    var_84 = var_75.visit(var_83)
    var_85 = len(var_73)
    assert var_85 == 0
    var_86 = []
    var_87 = lambda **kwargs: report_calls.append(kwargs)
    var_88 = module_0.Reachability(var_87)
    var_89 = module_1.Load()
    var_90 = module_1.Name()
    var_91 = module_1.withitem()
    var_92 = [var_91]
    var_93 = module_1.Pass()
    var_94 = [var_93]
    var_95 = module_1.AsyncWith()
    var_96 = var_88.visit(var_95)
    var_97 = len(var_86)
    assert var_97 == 0
    var_98 = []
    var_99 = lambda **kwargs: report_calls.append(kwargs)
    var_100 = module_0.Reachability(var_99)
    var_101 = False
    var_102 = module_1.Constant()
    var_103 = module_1.Pass()
    var_104 = [var_103]
    var_105 = []
    var_106 = module_1.While()
    var_107 = var_100.visit(var_106)
    var_108 = len(var_98)
    assert var_108 == 1
    var_109 = []
    var_110 = lambda **kwargs: report_calls.append(kwargs)
    var_111 = module_0.Reachability(var_110)
    var_112 = 'i'
    var_113 = module_1.Store()
    var_114 = module_1.Name()
    var_115 = module_1.Load()
    var_116 = module_1.Name()
    var_117 = module_1.Pass()
    var_118 = [var_117]
    var_119 = []
    var_120 = module_1.For()
    var_121 = var_111.visit(var_120)
    var_122 = len(var_109)
    assert var_122 == 0
    var_123 = []
    var_124 = lambda **kwargs: report_calls.append(kwargs)
    var_125 = module_0.Reachability(var_124)
    var_126 = module_1.Store()
    var_127 = module_1.Name()
    var_128 = module_1.Load()
    var_129 = module_1.Name()
    var_130 = module_1.Pass()
    var_131 = [var_130]
    var_132 = []
    var_133 = module_1.AsyncFor()
    var_134 = var_125.visit(var_133)
    var_135 = len(var_123)
    assert var_135 == 0
    var_136 = []
    var_137 = lambda **kwargs: report_calls.append(kwargs)
    var_138 = module_0.Reachability(var_137)
    var_139 = module_1.Constant()
    var_140 = module_1.Pass()
    var_141 = [var_140]
    var_142 = module_1.Pass()
    var_143 = [var_142]
    var_144 = module_1.If()
    var_145 = var_138.visit(var_144)
    var_146 = len(var_136)
    assert var_146 == 1
    var_147 = []
    var_148 = lambda **kwargs: report_calls.append(kwargs)
    var_149 = module_0.Reachability(var_148)
    var_150 = True
    var_151 = module_1.Constant()
    var_152 = module_1.Pass()
    var_153 = [var_152]
    var_154 = module_1.Pass()
    var_155 = [var_154]
    var_156 = module_1.If()
    var_157 = var_149.visit(var_156)
    var_158 = len(var_147)
    assert var_158 == 1
    var_159 = []
    var_160 = lambda **kwargs: report_calls.append(kwargs)
    var_161 = module_0.Reachability(var_160)
    var_162 = module_1.Constant()
    var_163 = module_1.Constant()
    var_164 = 2
    var_165 = module_1.Constant()
    var_166 = module_1.IfExp()
    var_167 = var_161.visit(var_166)
    var_168 = len(var_159)
    assert var_168 == 1
    var_169 = []
    var_170 = lambda **kwargs: report_calls.append(kwargs)
    var_171 = module_0.Reachability(var_170)
    var_172 = module_1.Pass()
    var_173 = [var_172]
    var_174 = 'Exception'
    var_175 = module_1.Load()
    var_176 = module_1.Name()
    var_177 = module_1.Pass()
    var_178 = [var_177]
    var_179 = module_1.ExceptHandler()
    var_180 = [var_179]
    var_181 = []
    var_182 = []
    var_183 = module_1.Try()
    var_184 = var_171.visit(var_183)
    var_185 = len(var_169)
    assert var_185 == 0



# Parsed testcases at query #18
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Pass()
    var_2 = module_0.Break()



# Parsed testcases at query #19
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Pass()
    var_2 = module_0.Break()



# Parsed testcases at query #20
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = module_1.Continue()
    var_6 = var_2.visit(var_5)
    var_7 = None
    var_8 = module_1.Return()
    var_9 = var_2.visit(var_8)
    var_10 = module_1.Raise()
    var_11 = var_2.visit(var_10)



# Parsed testcases at query #21
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = []
    var_6 = lambda **kwargs: report_calls.append(kwargs)
    var_7 = module_0.Reachability(var_6)
    var_8 = module_1.Continue()
    var_9 = var_7.visit(var_8)
    var_10 = []
    var_11 = lambda **kwargs: report_calls.append(kwargs)
    var_12 = module_0.Reachability(var_11)
    var_13 = module_1.Return()
    var_14 = var_12.visit(var_13)
    var_15 = []
    var_16 = lambda **kwargs: report_calls.append(kwargs)
    var_17 = module_0.Reachability(var_16)
    var_18 = module_1.Raise()
    var_19 = var_17.visit(var_18)
    var_20 = []
    var_21 = lambda **kwargs: report_calls.append(kwargs)
    var_22 = module_0.Reachability(var_21)
    var_23 = module_1.Return()
    var_24 = module_1.Pass()
    var_25 = [var_23, var_24]
    var_26 = []
    var_27 = module_1.Module()
    var_28 = var_22.visit(var_23)
    var_29 = var_22.visit(var_24)
    var_30 = var_22.visit(var_27)
    var_31 = len(var_20)
    assert var_31 == 1
    var_32 = []
    var_33 = lambda **kwargs: report_calls.append(kwargs)
    var_34 = module_0.Reachability(var_33)
    var_35 = False
    var_36 = module_1.Constant()
    var_37 = module_1.Pass()
    var_38 = [var_37]
    var_39 = []
    var_40 = module_1.While()
    var_41 = var_34.visit(var_40)
    var_42 = len(var_32)
    assert var_42 == 1
    var_43 = []
    var_44 = lambda **kwargs: report_calls.append(kwargs)
    var_45 = module_0.Reachability(var_44)
    var_46 = True
    var_47 = module_1.Constant()
    var_48 = module_1.Pass()
    var_49 = [var_48]
    var_50 = module_1.Pass()
    var_51 = [var_50]
    var_52 = module_1.If()
    var_53 = var_45.visit(var_52)
    var_54 = len(var_43)
    assert var_54 == 1
    var_55 = []
    var_56 = lambda **kwargs: report_calls.append(kwargs)
    var_57 = module_0.Reachability(var_56)
    var_58 = module_1.Constant()
    var_59 = module_1.Constant()
    var_60 = 2
    var_61 = module_1.Constant()
    var_62 = module_1.IfExp()
    var_63 = var_57.visit(var_62)
    var_64 = len(var_55)
    assert var_64 == 1
    var_65 = []
    var_66 = lambda **kwargs: report_calls.append(kwargs)
    var_67 = module_0.Reachability(var_66)
    var_68 = module_1.Return()
    var_69 = None
    var_70 = module_1.Pass()
    var_71 = [var_70]
    var_72 = module_1.ExceptHandler()
    var_73 = [var_68]
    var_74 = [var_72]
    var_75 = module_1.Pass()
    var_76 = [var_75]
    var_77 = []
    var_78 = module_1.Try()
    var_79 = var_67.visit(var_68)
    var_80 = var_67.visit(var_78)
    var_81 = len(var_65)
    assert var_81 == 1
    var_82 = []
    var_83 = lambda **kwargs: report_calls.append(kwargs)
    var_84 = module_0.Reachability(var_83)
    var_85 = module_1.Pass()
    var_86 = var_84.visit(var_85)



# Parsed testcases at query #22
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Pass()
    var_2 = module_0.Break()



# Parsed testcases at query #23
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = var_2.reset()
    var_4 = var_2.reset()
    var_5 = module_1.Break()
    var_6 = None
    var_7 = module_1.Return()
    var_8 = [var_5, var_7]
    var_9 = module_1.Module()
    var_10 = var_2.visit(var_9)
    var_11 = len(var_0)
    assert var_11 == 1
    var_12 = var_2.reset()
    var_13 = False
    var_14 = module_1.Constant()
    var_15 = module_1.Pass()
    var_16 = [var_15]
    var_17 = []
    var_18 = module_1.While()
    var_19 = var_2.visit(var_18)
    var_20 = len(var_0)
    assert var_20 == 1
    var_21 = var_2.reset()
    var_22 = True
    var_23 = module_1.Constant()
    var_24 = module_1.Pass()
    var_25 = [var_24]
    var_26 = []
    var_27 = module_1.While()
    var_28 = var_2.visit(var_27)
    var_29 = var_2.reset()
    var_30 = module_1.Constant()
    var_31 = module_1.Pass()
    var_32 = [var_31]
    var_33 = module_1.Pass()
    var_34 = [var_33]
    var_35 = module_1.If()
    var_36 = var_2.visit(var_35)
    var_37 = len(var_0)
    assert var_37 == 1
    var_38 = var_2.reset()
    var_39 = module_1.Constant()
    var_40 = module_1.Pass()
    var_41 = [var_40]
    var_42 = []
    var_43 = module_1.If()
    var_44 = var_2.visit(var_43)
    var_45 = len(var_0)
    assert var_45 == 1
    var_46 = var_2.reset()
    var_47 = module_1.Constant()
    var_48 = module_1.Constant()
    var_49 = 2
    var_50 = module_1.Constant()
    var_51 = module_1.IfExp()
    var_52 = var_2.visit(var_51)
    var_53 = len(var_0)
    assert var_53 == 1
    var_54 = var_2.reset()
    var_55 = module_1.Return()
    var_56 = [var_55]
    var_57 = module_1.Pass()
    var_58 = [var_57]
    var_59 = module_1.ExceptHandler()
    var_60 = [var_59]
    var_61 = []
    var_62 = []
    var_63 = module_1.Try()
    var_64 = var_2.visit(var_63)
    var_65 = var_2.reset()
    var_66 = 'test_func'
    var_67 = []
    var_68 = []
    var_69 = []
    var_70 = []
    var_71 = []
    var_72 = module_1.arguments(*var_68)
    var_73 = module_1.Return()
    var_74 = module_1.Pass()
    var_75 = [var_73, var_74]
    var_76 = []
    var_77 = module_1.FunctionDef(*var_72)
    var_78 = var_2.visit(var_77)
    var_79 = len(var_0)
    assert var_79 == 1



# Parsed testcases at query #24
#--------------------------


import vulture.reachability as module_0

def test_case_0():
    var_0 = 'Test Reachability constructor initializes correctly.'
    var_1 = None
    var_2 = lambda name, first_node, last_node=None, message='': var_1
    var_3 = module_0.Reachability(var_2)
    var_4 = set()



# Parsed testcases at query #25
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = []
    var_6 = lambda **kwargs: report_calls.append(kwargs)
    var_7 = module_0.Reachability(var_6)
    var_8 = module_1.Continue()
    var_9 = var_7.visit(var_8)
    var_10 = []
    var_11 = lambda **kwargs: report_calls.append(kwargs)
    var_12 = module_0.Reachability(var_11)
    var_13 = 1
    var_14 = module_1.Constant()
    var_15 = module_1.Return()
    var_16 = var_12.visit(var_15)
    var_17 = []
    var_18 = lambda **kwargs: report_calls.append(kwargs)
    var_19 = module_0.Reachability(var_18)
    var_20 = module_1.Raise()
    var_21 = var_19.visit(var_20)
    var_22 = []
    var_23 = lambda **kwargs: report_calls.append(kwargs)
    var_24 = module_0.Reachability(var_23)
    var_25 = module_1.Pass()
    var_26 = module_1.Pass()
    var_27 = [var_25, var_26]
    var_28 = []
    var_29 = module_1.Module()
    var_30 = var_24.visit(var_29)
    var_31 = len(var_22)
    assert var_31 == 0
    var_32 = []
    var_33 = lambda **kwargs: report_calls.append(kwargs)
    var_34 = module_0.Reachability(var_33)
    var_35 = module_1.Constant()
    var_36 = module_1.Return()
    var_37 = module_1.Pass()
    var_38 = [var_36, var_37]
    var_39 = []
    var_40 = module_1.Module()
    var_41 = var_34.visit(var_36)
    var_42 = var_34.visit(var_40)
    var_43 = len(var_32)
    assert var_43 == 1
    var_44 = []
    var_45 = lambda **kwargs: report_calls.append(kwargs)
    var_46 = module_0.Reachability(var_45)
    var_47 = False
    var_48 = module_1.Constant()
    var_49 = module_1.Pass()
    var_50 = [var_49]
    var_51 = []
    var_52 = module_1.If()
    var_53 = var_46.visit(var_52)
    var_54 = len(var_44)
    assert var_54 == 1
    var_55 = []
    var_56 = lambda **kwargs: report_calls.append(kwargs)
    var_57 = module_0.Reachability(var_56)
    var_58 = True
    var_59 = module_1.Constant()
    var_60 = module_1.Pass()
    var_61 = [var_60]
    var_62 = module_1.Pass()
    var_63 = [var_62]
    var_64 = module_1.If()
    var_65 = var_57.visit(var_64)
    var_66 = len(var_55)
    assert var_66 == 1
    var_67 = []
    var_68 = lambda **kwargs: report_calls.append(kwargs)
    var_69 = module_0.Reachability(var_68)
    var_70 = True
    var_71 = module_1.Constant()
    var_72 = module_1.Pass()
    var_73 = [var_72]
    var_74 = []
    var_75 = module_1.If()
    var_76 = var_69.visit(var_75)
    var_77 = len(var_67)
    assert var_77 == 1
    var_78 = []
    var_79 = lambda **kwargs: report_calls.append(kwargs)
    var_80 = module_0.Reachability(var_79)
    var_81 = module_1.Constant()
    var_82 = module_1.Constant()
    var_83 = 2
    var_84 = module_1.Constant()
    var_85 = module_1.IfExp()
    var_86 = var_80.visit(var_85)
    var_87 = len(var_78)
    assert var_87 == 1
    var_88 = []
    var_89 = lambda **kwargs: report_calls.append(kwargs)
    var_90 = module_0.Reachability(var_89)
    var_91 = True
    var_92 = module_1.Constant()
    var_93 = module_1.Constant()
    var_94 = module_1.Constant()
    var_95 = module_1.IfExp()
    var_96 = var_90.visit(var_95)
    var_97 = len(var_88)
    assert var_97 == 1
    var_98 = []
    var_99 = lambda **kwargs: report_calls.append(kwargs)
    var_100 = module_0.Reachability(var_99)
    var_101 = module_1.Constant()
    var_102 = module_1.Pass()
    var_103 = [var_102]
    var_104 = []
    var_105 = module_1.While()
    var_106 = var_100.visit(var_105)
    var_107 = len(var_98)
    assert var_107 == 1
    var_108 = []
    var_109 = lambda **kwargs: report_calls.append(kwargs)
    var_110 = module_0.Reachability(var_109)
    var_111 = True
    var_112 = module_1.Constant()
    var_113 = module_1.Pass()
    var_114 = [var_113]
    var_115 = []
    var_116 = module_1.While()
    var_117 = var_110.visit(var_116)
    var_118 = []
    var_119 = lambda **kwargs: report_calls.append(kwargs)
    var_120 = module_0.Reachability(var_119)
    var_121 = True
    var_122 = module_1.Constant()
    var_123 = module_1.Break()
    var_124 = [var_123]
    var_125 = []
    var_126 = module_1.While()
    var_127 = var_120.visit(var_126)
    var_128 = []
    var_129 = lambda **kwargs: report_calls.append(kwargs)
    var_130 = module_0.Reachability(var_129)
    var_131 = True
    var_132 = module_1.Constant()
    var_133 = module_1.Pass()
    var_134 = [var_133]
    var_135 = module_1.Pass()
    var_136 = [var_135]
    var_137 = module_1.While()
    var_138 = var_130.visit(var_137)
    var_139 = len(var_128)
    assert var_139 == 1
    var_140 = []
    var_141 = lambda **kwargs: report_calls.append(kwargs)
    var_142 = module_0.Reachability(var_141)
    var_143 = module_1.Pass()
    var_144 = [var_143]
    var_145 = 'Exception'
    var_146 = module_1.Load()
    var_147 = module_1.Name()
    var_148 = None
    var_149 = module_1.Pass()
    var_150 = [var_149]
    var_151 = module_1.ExceptHandler()
    var_152 = [var_151]
    var_153 = []
    var_154 = []
    var_155 = module_1.Try()
    var_156 = var_142.visit(var_155)
    var_157 = len(var_140)
    assert var_157 == 0
    var_158 = []
    var_159 = lambda **kwargs: report_calls.append(kwargs)
    var_160 = module_0.Reachability(var_159)
    var_161 = module_1.Constant()
    var_162 = module_1.Return()
    var_163 = [var_162]
    var_164 = module_1.Load()
    var_165 = module_1.Name()
    var_166 = module_1.Pass()
    var_167 = [var_166]
    var_168 = module_1.ExceptHandler()
    var_169 = [var_168]
    var_170 = module_1.Pass()
    var_171 = [var_170]
    var_172 = []
    var_173 = module_1.Try()
    var_174 = var_160.visit(var_162)
    var_175 = var_160.visit(var_173)
    var_176 = len(var_158)
    assert var_176 == 1
    var_177 = []
    var_178 = lambda **kwargs: report_calls.append(kwargs)
    var_179 = module_0.Reachability(var_178)
    var_180 = module_1.Break()
    var_181 = var_179.visit(var_180)
    var_182 = var_179._no_fall_through_nodes
    var_183 = len(var_182)
    assert var_183 == 1
    var_184 = var_179.reset()
    var_185 = var_179._no_fall_through_nodes
    var_186 = len(var_185)
    assert var_186 == 0



# Parsed testcases at query #26
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = len(var_0)
    assert var_5 == 0
    var_6 = var_2.reset()
    var_7 = module_1.Continue()
    var_8 = var_2.visit(var_7)
    var_9 = var_2.reset()
    var_10 = None
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = True
    var_18 = module_1.Constant()
    var_19 = module_1.Pass()
    var_20 = [var_19]
    var_21 = []
    var_22 = module_1.If()
    var_23 = var_2.visit(var_22)
    var_24 = len(var_0)
    var_25 = var_2.reset()
    var_26 = module_1.Pass()
    var_27 = [var_26]
    var_28 = []
    var_29 = module_1.Module()
    var_30 = var_2.visit(var_29)
    var_31 = len(var_0)
    assert var_31 == 0
    var_32 = var_2.reset()
    var_33 = False
    var_34 = module_1.Constant()
    var_35 = module_1.Pass()
    var_36 = [var_35]
    var_37 = []
    var_38 = module_1.While()
    var_39 = var_2.visit(var_38)
    var_40 = len(var_0)



# Parsed testcases at query #27
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = 'x = 1'
    var_3 = module_0.parse(var_2)
    var_4 = len(var_0)
    assert var_4 == 0
    var_5 = set()



# Parsed testcases at query #28
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Break()
    var_2 = module_0.Continue()
    var_3 = module_0.Return()



# Parsed testcases at query #29
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.Pass()
    var_2 = module_0.Break()
    var_3 = len(var_0)
    assert var_3 == 0



# Parsed testcases at query #30
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = None
    var_10 = module_1.Return()
    var_11 = var_2.visit(var_10)
    var_12 = var_2.reset()
    var_13 = module_1.Raise()
    var_14 = var_2.visit(var_13)
    var_15 = var_2.reset()
    var_16 = module_1.Return()
    var_17 = module_1.Pass()
    var_18 = [var_16, var_17]
    var_19 = []
    var_20 = module_1.Module()
    var_21 = var_2.visit(var_16)
    var_22 = var_2.visit(var_17)
    var_23 = len(var_0)
    assert var_23 == 1
    var_24 = var_2.reset()
    var_25 = False
    var_26 = module_1.Constant()
    var_27 = module_1.Pass()
    var_28 = [var_27]
    var_29 = []
    var_30 = module_1.While()
    var_31 = var_2.visit(var_30)
    var_32 = len(var_0)
    assert var_32 == 1
    var_33 = var_2.reset()
    var_34 = True
    var_35 = module_1.Constant()
    var_36 = module_1.Break()
    var_37 = [var_36]
    var_38 = []
    var_39 = module_1.While()
    var_40 = var_2.visit(var_39)
    var_41 = 'message'
    var_42 = "unreachable code after 'break'"
    var_43 = var_2.reset()
    var_44 = module_1.Constant()
    var_45 = module_1.Pass()
    var_46 = [var_45]
    var_47 = module_1.Pass()
    var_48 = [var_47]
    var_49 = module_1.If()
    var_50 = var_2.visit(var_49)
    var_51 = len(var_0)
    assert var_51 == 1
    var_52 = var_2.reset()
    var_53 = module_1.Constant()
    var_54 = module_1.Pass()
    var_55 = [var_54]
    var_56 = module_1.Pass()
    var_57 = [var_56]
    var_58 = module_1.If()
    var_59 = var_2.visit(var_58)
    var_60 = len(var_0)
    assert var_60 == 1
    var_61 = var_2.reset()
    var_62 = module_1.Constant()
    var_63 = module_1.Constant()
    var_64 = 2
    var_65 = module_1.Constant()
    var_66 = module_1.IfExp()
    var_67 = var_2.visit(var_66)
    var_68 = len(var_0)
    assert var_68 == 1
    var_69 = var_2.reset()
    var_70 = module_1.Return()
    var_71 = [var_70]
    var_72 = module_1.Pass()
    var_73 = [var_72]
    var_74 = module_1.ExceptHandler()
    var_75 = [var_74]
    var_76 = module_1.Pass()
    var_77 = [var_76]
    var_78 = []
    var_79 = module_1.Try()
    var_80 = var_79.body[var_25]
    var_81 = var_2.visit(var_80)
    var_82 = var_79.handlers[var_25]
    var_83 = var_82.body[var_25]
    var_84 = var_2.visit(var_83)
    var_85 = len(var_0)
    assert var_85 == 1



# Parsed testcases at query #31
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = None
    var_10 = module_1.Constant()
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = 1
    var_18 = module_1.Constant()
    var_19 = module_1.Return()
    var_20 = 2
    var_21 = module_1.Constant()
    var_22 = module_1.Expr()
    var_23 = [var_19, var_22]
    var_24 = []
    var_25 = module_1.Module()
    var_26 = var_2.visit(var_25)
    var_27 = len(var_0)
    assert var_27 == 1
    var_28 = var_2.reset()
    var_29 = False
    var_30 = module_1.Constant()
    var_31 = module_1.Constant()
    var_32 = module_1.Expr()
    var_33 = [var_32]
    var_34 = []
    var_35 = module_1.If()
    var_36 = var_2.visit(var_35)
    var_37 = len(var_0)
    assert var_37 == 1
    var_38 = var_2.reset()
    var_39 = True
    var_40 = module_1.Constant()
    var_41 = module_1.Constant()
    var_42 = module_1.Expr()
    var_43 = [var_42]
    var_44 = module_1.Constant()
    var_45 = module_1.Expr()
    var_46 = [var_45]
    var_47 = module_1.If()
    var_48 = var_2.visit(var_47)
    var_49 = len(var_0)
    assert var_49 == 1
    var_50 = var_2.reset()
    var_51 = True
    var_52 = module_1.Constant()
    var_53 = module_1.Constant()
    var_54 = module_1.Expr()
    var_55 = [var_54]
    var_56 = []
    var_57 = module_1.If()
    var_58 = var_2.visit(var_57)
    var_59 = len(var_0)
    assert var_59 == 1
    var_60 = var_2.reset()
    var_61 = module_1.Constant()
    var_62 = module_1.Constant()
    var_63 = module_1.Expr()
    var_64 = [var_63]
    var_65 = []
    var_66 = module_1.While()
    var_67 = var_2.visit(var_66)
    var_68 = len(var_0)
    assert var_68 == 1
    var_69 = var_2.reset()
    var_70 = True
    var_71 = module_1.Constant()
    var_72 = module_1.Break()
    var_73 = [var_72]
    var_74 = []
    var_75 = module_1.While()
    var_76 = var_2.visit(var_75)
    var_77 = len(var_0)
    assert var_77 == 0
    var_78 = var_2.reset()
    var_79 = 'x'
    var_80 = module_1.Store()
    var_81 = module_1.Name()
    var_82 = module_1.Constant()
    var_83 = [var_82]
    var_84 = module_1.Load()
    var_85 = module_1.List()
    var_86 = module_1.Constant()
    var_87 = module_1.Expr()
    var_88 = [var_87]
    var_89 = []
    var_90 = module_1.For()
    var_91 = var_2.visit(var_90)
    var_92 = len(var_0)
    assert var_92 == 0
    var_93 = var_2.reset()
    var_94 = module_1.Constant()
    var_95 = module_1.Return()
    var_96 = [var_95]
    var_97 = module_1.Constant()
    var_98 = module_1.Expr()
    var_99 = [var_98]
    var_100 = module_1.ExceptHandler()
    var_101 = [var_100]
    var_102 = 3
    var_103 = module_1.Constant()
    var_104 = module_1.Expr()
    var_105 = [var_104]
    var_106 = []
    var_107 = module_1.Try()
    var_108 = var_2.visit(var_107)
    var_109 = len(var_0)
    assert var_109 == 1
    var_110 = var_2.reset()
    var_111 = module_1.Constant()
    var_112 = module_1.Return()
    var_113 = [var_112]
    var_114 = module_1.Constant()
    var_115 = module_1.Return()
    var_116 = [var_115]
    var_117 = module_1.ExceptHandler()
    var_118 = [var_117]
    var_119 = []
    var_120 = []
    var_121 = module_1.Try()
    var_122 = var_2.visit(var_121)
    var_123 = var_2.reset()
    var_124 = module_1.Constant()
    var_125 = module_1.Constant()
    var_126 = module_1.Constant()
    var_127 = module_1.IfExp()
    var_128 = var_2.visit(var_127)
    var_129 = len(var_0)
    assert var_129 == 1
    var_130 = var_2.reset()
    var_131 = True
    var_132 = module_1.Constant()
    var_133 = module_1.Constant()
    var_134 = module_1.Constant()
    var_135 = module_1.IfExp()
    var_136 = var_2.visit(var_135)
    var_137 = len(var_0)
    assert var_137 == 1
    var_138 = var_2.reset()
    var_139 = 'test_func'
    var_140 = []
    var_141 = []
    var_142 = []
    var_143 = []
    var_144 = []
    var_145 = module_1.arguments(*var_141)
    var_146 = module_1.Constant()
    var_147 = module_1.Return()
    var_148 = module_1.Constant()
    var_149 = module_1.Expr()
    var_150 = [var_147, var_149]
    var_151 = []
    var_152 = module_1.FunctionDef(*var_145)
    var_153 = var_2.visit(var_152)
    var_154 = len(var_0)
    assert var_154 == 1



# Parsed testcases at query #32
#--------------------------


import vulture.reachability as module_0
import ast as module_1

def test_case_0():
    var_0 = []
    var_1 = lambda **kwargs: report_calls.append(kwargs)
    var_2 = module_0.Reachability(var_1)
    var_3 = module_1.Break()
    var_4 = var_2.visit(var_3)
    var_5 = var_2.reset()
    var_6 = module_1.Continue()
    var_7 = var_2.visit(var_6)
    var_8 = var_2.reset()
    var_9 = None
    var_10 = module_1.Constant()
    var_11 = module_1.Return()
    var_12 = var_2.visit(var_11)
    var_13 = var_2.reset()
    var_14 = module_1.Raise()
    var_15 = var_2.visit(var_14)
    var_16 = var_2.reset()
    var_17 = 1
    var_18 = module_1.Constant()
    var_19 = module_1.Return()
    var_20 = 2
    var_21 = module_1.Constant()
    var_22 = module_1.Expr()
    var_23 = [var_19, var_22]
    var_24 = []
    var_25 = module_1.Module()
    var_26 = var_2.visit(var_19)
    var_27 = var_2.visit(var_22)
    var_28 = var_2.visit(var_25)
    var_29 = len(var_0)
    var_30 = 'message'
    var_31 = -1
    var_32 = var_0[var_31][var_30]
    var_33 = 'unreachable code after'
    var_34 = var_2.reset()
    var_35 = False
    var_36 = module_1.Constant()
    var_37 = module_1.Pass()
    var_38 = [var_37]
    var_39 = []
    var_40 = module_1.While()
    var_41 = var_2.visit(var_40)
    var_42 = "unsatisfiable 'while' condition"
    var_43 = var_2.reset()
    var_44 = True
    var_45 = module_1.Constant()
    var_46 = module_1.Pass()
    var_47 = [var_46]
    var_48 = []
    var_49 = module_1.While()
    var_50 = var_2.visit(var_49)
    var_51 = var_2.reset()
    var_52 = True
    var_53 = module_1.Constant()
    var_54 = module_1.Pass()
    var_55 = [var_54]
    var_56 = module_1.Pass()
    var_57 = [var_56]
    var_58 = module_1.If()
    var_59 = var_2.visit(var_58)
    var_60 = "unreachable 'else' block"
    var_61 = var_2.reset()
    var_62 = module_1.Constant()
    var_63 = module_1.Constant()
    var_64 = module_1.Constant()
    var_65 = module_1.IfExp()
    var_66 = var_2.visit(var_65)
    var_67 = "unsatisfiable 'ternary' condition"
    var_68 = var_2.reset()
    var_69 = module_1.Constant()
    var_70 = module_1.Return()
    var_71 = [var_70]
    var_72 = 'Exception'
    var_73 = module_1.Load()
    var_74 = module_1.Name()
    var_75 = module_1.Pass()
    var_76 = [var_75]
    var_77 = module_1.ExceptHandler()
    var_78 = [var_77]
    var_79 = module_1.Pass()
    var_80 = [var_79]
    var_81 = []
    var_82 = module_1.Try()
    var_83 = var_82.body[var_35]
    var_84 = var_2.visit(var_83)
    var_85 = var_2.visit(var_82)
    var_86 = var_2.reset()
    var_87 = 'test_func'
    var_88 = []
    var_89 = []
    var_90 = []
    var_91 = []
    var_92 = []
    var_93 = module_1.arguments(*var_89)
    var_94 = module_1.Pass()
    var_95 = [var_94]
    var_96 = []
    var_97 = module_1.FunctionDef(*var_93)
    var_98 = var_2.visit(var_97)
    var_99 = len(var_0)
    assert var_99 == 0
    var_100 = var_2.reset()
    var_101 = 'i'
    var_102 = module_1.Store()
    var_103 = module_1.Name()
    var_104 = module_1.Constant()
    var_105 = [var_104]
    var_106 = module_1.Load()
    var_107 = module_1.List()
    var_108 = module_1.Pass()
    var_109 = [var_108]
    var_110 = []
    var_111 = module_1.For()
    var_112 = var_2.visit(var_111)
    var_113 = len(var_0)
    assert var_113 == 0
    var_114 = var_2.reset()
    var_115 = 'open'
    var_116 = module_1.Load()
    var_117 = module_1.Name()
    var_118 = 'file.txt'
    var_119 = module_1.Constant()
    var_120 = [var_119]
    var_121 = []
    var_122 = module_1.Call(*var_120)
    var_123 = module_1.withitem()
    var_124 = [var_123]
    var_125 = module_1.Pass()
    var_126 = [var_125]
    var_127 = module_1.With()
    var_128 = var_2.visit(var_127)
    var_129 = len(var_0)
    assert var_129 == 0
    var_130 = var_2.reset()
    var_131 = module_1.Break()
    var_132 = module_1.Constant()
    var_133 = module_1.Expr()
    var_134 = True
    var_135 = module_1.Constant()
    var_136 = [var_131, var_133]
    var_137 = []
    var_138 = module_1.If()
    var_139 = var_2.visit(var_131)
    var_140 = var_2.visit(var_138)
    var_141 = "unreachable code after 'break'"



# Parsed testcases at query #33
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = 'some_node'
    var_3 = set()



# Parsed testcases at query #34
#--------------------------


def test_case_0():
    var_0 = []
    var_1 = set()



# Parsed testcases at query #35
#--------------------------


import ast as module_0

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = module_0.Pass()
    var_3 = set()



