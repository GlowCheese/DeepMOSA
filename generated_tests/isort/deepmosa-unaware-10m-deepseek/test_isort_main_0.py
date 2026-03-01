# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.main as module_0
import tokenize as module_1
import email._header_value_parser as module_2
import posixpath as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort_imports(var_0, var_0, ask_to_apply=var_0, write_to_stdout=var_0)

def test_case_1():
    var_0 = module_1.group()
    assert var_0 == '()'
    assert module_1.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_1.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_1.ENDMARKER == 0
    assert module_1.NAME == 1
    assert module_1.NUMBER == 2
    assert module_1.STRING == 3
    assert module_1.NEWLINE == 4
    assert module_1.INDENT == 5
    assert module_1.DEDENT == 6
    assert module_1.LPAR == 7
    assert module_1.RPAR == 8
    assert module_1.LSQB == 9
    assert module_1.RSQB == 10
    assert module_1.COLON == 11
    assert module_1.COMMA == 12
    assert module_1.SEMI == 13
    assert module_1.PLUS == 14
    assert module_1.MINUS == 15
    assert module_1.STAR == 16
    assert module_1.SLASH == 17
    assert module_1.VBAR == 18
    assert module_1.AMPER == 19
    assert module_1.LESS == 20
    assert module_1.GREATER == 21
    assert module_1.EQUAL == 22
    assert module_1.DOT == 23
    assert module_1.PERCENT == 24
    assert module_1.LBRACE == 25
    assert module_1.RBRACE == 26
    assert module_1.EQEQUAL == 27
    assert module_1.NOTEQUAL == 28
    assert module_1.LESSEQUAL == 29
    assert module_1.GREATEREQUAL == 30
    assert module_1.TILDE == 31
    assert module_1.CIRCUMFLEX == 32
    assert module_1.LEFTSHIFT == 33
    assert module_1.RIGHTSHIFT == 34
    assert module_1.DOUBLESTAR == 35
    assert module_1.PLUSEQUAL == 36
    assert module_1.MINEQUAL == 37
    assert module_1.STAREQUAL == 38
    assert module_1.SLASHEQUAL == 39
    assert module_1.PERCENTEQUAL == 40
    assert module_1.AMPEREQUAL == 41
    assert module_1.VBAREQUAL == 42
    assert module_1.CIRCUMFLEXEQUAL == 43
    assert module_1.LEFTSHIFTEQUAL == 44
    assert module_1.RIGHTSHIFTEQUAL == 45
    assert module_1.DOUBLESTAREQUAL == 46
    assert module_1.DOUBLESLASH == 47
    assert module_1.DOUBLESLASHEQUAL == 48
    assert module_1.AT == 49
    assert module_1.ATEQUAL == 50
    assert module_1.RARROW == 51
    assert module_1.ELLIPSIS == 52
    assert module_1.COLONEQUAL == 53
    assert module_1.OP == 54
    assert module_1.AWAIT == 55
    assert module_1.ASYNC == 56
    assert module_1.TYPE_IGNORE == 57
    assert module_1.TYPE_COMMENT == 58
    assert module_1.SOFT_KEYWORD == 59
    assert module_1.ERRORTOKEN == 60
    assert module_1.COMMENT == 61
    assert module_1.NL == 62
    assert module_1.ENCODING == 63
    assert module_1.N_TOKENS == 64
    assert module_1.NT_OFFSET == 256
    assert module_1.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_1.cookie_re).__module__}.{type(module_1.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.blank_re).__module__}.{type(module_1.blank_re).__qualname__}' == 're.Pattern'
    assert module_1.Whitespace == '[ \\f\\t]*'
    assert module_1.Comment == '#[^\\r\\n]*'
    assert module_1.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_1.Name == '\\w+'
    assert module_1.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_1.Binnumber == '0[bB](?:_?[01])+'
    assert module_1.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_1.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_1.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_1.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_1.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_1.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_1.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_1.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_1.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_1.StringPrefix == '(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)'
    assert module_1.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_1.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_1.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_1.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_1.Triple == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)""")'
    assert module_1.String == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_1.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_1.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_1.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_1.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_1.ContStr == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_1.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"""))'
    assert module_1.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_1.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_1.single_quoted == {"fR'", "Br'", 'Rb"', 'rB"', 'b"', 'rb"', 'Br"', "f'", 'fR"', 'fr"', 'u"', "Rf'", "R'", "'", 'R"', 'Fr"', 'BR"', "F'", "rb'", "r'", "Fr'", 'r"', "br'", "RF'", "RB'", "BR'", 'rF"', 'RF"', "rf'", 'f"', "FR'", 'FR"', "u'", 'bR"', 'F"', 'U"', 'Rf"', "Rb'", 'br"', 'RB"', "rF'", '"', "b'", "U'", "fr'", "B'", 'rf"', "rB'", 'B"', "bR'"}
    assert module_1.triple_quoted == {"R'''", 'rb"""', 'Fr"""', 'r"""', "F'''", 'R"""', "u'''", "Br'''", 'FR"""', 'rf"""', "U'''", 'br"""', "rf'''", "RF'''", 'Br"""', 'rF"""', "br'''", 'Rf"""', "'''", "rb'''", 'B"""', 'RF"""', "FR'''", "bR'''", 'F"""', "b'''", "r'''", 'bR"""', 'U"""', "RB'''", "BR'''", 'rB"""', "fr'''", "rB'''", "Rb'''", 'BR"""', 'fR"""', "Fr'''", 'RB"""', 'Rb"""', '"""', "Rf'''", 'b"""', 'f"""', 'fr"""', "fR'''", "B'''", "rF'''", "f'''", 'u"""'}
    assert module_1.t == 'B'
    assert module_1.u == "B'''"
    assert module_1.tabsize == 8
    var_1 = module_0.parse_args(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_2 = module_0.sort_imports(var_0, var_0, var_0)
    var_3 = var_0.__str__()
    assert var_3 == '()'
    var_4 = var_0.__lt__(var_3)
    assert var_4 is False

def test_case_2():
    var_0 = module_1.group()
    assert var_0 == '()'
    assert module_1.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_1.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_1.ENDMARKER == 0
    assert module_1.NAME == 1
    assert module_1.NUMBER == 2
    assert module_1.STRING == 3
    assert module_1.NEWLINE == 4
    assert module_1.INDENT == 5
    assert module_1.DEDENT == 6
    assert module_1.LPAR == 7
    assert module_1.RPAR == 8
    assert module_1.LSQB == 9
    assert module_1.RSQB == 10
    assert module_1.COLON == 11
    assert module_1.COMMA == 12
    assert module_1.SEMI == 13
    assert module_1.PLUS == 14
    assert module_1.MINUS == 15
    assert module_1.STAR == 16
    assert module_1.SLASH == 17
    assert module_1.VBAR == 18
    assert module_1.AMPER == 19
    assert module_1.LESS == 20
    assert module_1.GREATER == 21
    assert module_1.EQUAL == 22
    assert module_1.DOT == 23
    assert module_1.PERCENT == 24
    assert module_1.LBRACE == 25
    assert module_1.RBRACE == 26
    assert module_1.EQEQUAL == 27
    assert module_1.NOTEQUAL == 28
    assert module_1.LESSEQUAL == 29
    assert module_1.GREATEREQUAL == 30
    assert module_1.TILDE == 31
    assert module_1.CIRCUMFLEX == 32
    assert module_1.LEFTSHIFT == 33
    assert module_1.RIGHTSHIFT == 34
    assert module_1.DOUBLESTAR == 35
    assert module_1.PLUSEQUAL == 36
    assert module_1.MINEQUAL == 37
    assert module_1.STAREQUAL == 38
    assert module_1.SLASHEQUAL == 39
    assert module_1.PERCENTEQUAL == 40
    assert module_1.AMPEREQUAL == 41
    assert module_1.VBAREQUAL == 42
    assert module_1.CIRCUMFLEXEQUAL == 43
    assert module_1.LEFTSHIFTEQUAL == 44
    assert module_1.RIGHTSHIFTEQUAL == 45
    assert module_1.DOUBLESTAREQUAL == 46
    assert module_1.DOUBLESLASH == 47
    assert module_1.DOUBLESLASHEQUAL == 48
    assert module_1.AT == 49
    assert module_1.ATEQUAL == 50
    assert module_1.RARROW == 51
    assert module_1.ELLIPSIS == 52
    assert module_1.COLONEQUAL == 53
    assert module_1.OP == 54
    assert module_1.AWAIT == 55
    assert module_1.ASYNC == 56
    assert module_1.TYPE_IGNORE == 57
    assert module_1.TYPE_COMMENT == 58
    assert module_1.SOFT_KEYWORD == 59
    assert module_1.ERRORTOKEN == 60
    assert module_1.COMMENT == 61
    assert module_1.NL == 62
    assert module_1.ENCODING == 63
    assert module_1.N_TOKENS == 64
    assert module_1.NT_OFFSET == 256
    assert module_1.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_1.cookie_re).__module__}.{type(module_1.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.blank_re).__module__}.{type(module_1.blank_re).__qualname__}' == 're.Pattern'
    assert module_1.Whitespace == '[ \\f\\t]*'
    assert module_1.Comment == '#[^\\r\\n]*'
    assert module_1.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_1.Name == '\\w+'
    assert module_1.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_1.Binnumber == '0[bB](?:_?[01])+'
    assert module_1.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_1.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_1.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_1.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_1.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_1.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_1.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_1.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_1.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_1.StringPrefix == '(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)'
    assert module_1.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_1.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_1.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_1.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_1.Triple == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)""")'
    assert module_1.String == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_1.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_1.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_1.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_1.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_1.ContStr == '((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_1.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"""))'
    assert module_1.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'\'\'|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|b|U|rb|rB|R|fR|Rf|rf|bR|RB|Fr|RF|BR|Rb|rF|u|r|F|Br|br|fr|f|FR|B)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_1.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_1.single_quoted == {"fR'", "Br'", 'Rb"', 'rB"', 'b"', 'rb"', 'Br"', "f'", 'fR"', 'fr"', 'u"', "Rf'", "R'", "'", 'R"', 'Fr"', 'BR"', "F'", "rb'", "r'", "Fr'", 'r"', "br'", "RF'", "RB'", "BR'", 'rF"', 'RF"', "rf'", 'f"', "FR'", 'FR"', "u'", 'bR"', 'F"', 'U"', 'Rf"', "Rb'", 'br"', 'RB"', "rF'", '"', "b'", "U'", "fr'", "B'", 'rf"', "rB'", 'B"', "bR'"}
    assert module_1.triple_quoted == {"R'''", 'rb"""', 'Fr"""', 'r"""', "F'''", 'R"""', "u'''", "Br'''", 'FR"""', 'rf"""', "U'''", 'br"""', "rf'''", "RF'''", 'Br"""', 'rF"""', "br'''", 'Rf"""', "'''", "rb'''", 'B"""', 'RF"""', "FR'''", "bR'''", 'F"""', "b'''", "r'''", 'bR"""', 'U"""', "RB'''", "BR'''", 'rB"""', "fr'''", "rB'''", "Rb'''", 'BR"""', 'fR"""', "Fr'''", 'RB"""', 'Rb"""', '"""', "Rf'''", 'b"""', 'f"""', 'fr"""', "fR'''", "B'''", "rF'''", "f'''", 'u"""'}
    assert module_1.t == 'B'
    assert module_1.u == "B'''"
    assert module_1.tabsize == 8
    var_1 = module_0.parse_args(var_0)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

def test_case_3():
    var_0 = False
    var_1 = module_0.SortAttempt(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.main.SortAttempt'
    assert var_1.incorrectly_sorted is False
    assert var_1.skipped is False
    assert var_1.supported_encoding is False
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = True
    module_0.sort_imports(var_0, var_0, var_1, write_to_stdout=var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = True
    var_1 = True
    var_2 = None
    module_0.sort_imports(var_0, var_2, var_1)

def test_case_6():
    var_0 = None
    var_1 = module_2.get_invalid_parameter(var_0)
    assert module_2.hexdigits == '0123456789abcdefABCDEF'
    assert module_2.WSP == {'\t', ' '}
    assert module_2.CFWS_LEADER == {'(', '\t', ' '}
    assert module_2.SPECIALS == {'.', '<', ',', '(', '[', ']', '>', '\\', ')', '"', ';', '@', ':'}
    assert module_2.ATOM_ENDS == {'.', '<', '(', ',', '[', ']', '>', '\\', ')', '"', ':', ';', '@', '\t', ' '}
    assert module_2.DOT_ATOM_ENDS == {',', '<', '(', '[', ']', '>', '\\', ')', '"', ':', ';', '@', '\t', ' '}
    assert module_2.PHRASE_ENDS == {'@', '<', ',', '[', ']', '\\', ')', ';', '>', ':'}
    assert module_2.TSPECIALS == {'/', '?', '<', '(', '[', ',', ']', '>', '\\', ')', '"', '=', ';', '@', ':'}
    assert module_2.TOKEN_ENDS == {'/', '?', '@', '<', '(', '[', ',', ']', '\\', ')', '"', '=', ':', ';', '>', '\t', ' '}
    assert module_2.ASPECIALS == {'/', '?', '@', "'", '<', '(', '[', ',', '*', ']', '%', '\\', ')', '"', '=', ';', '>', ':'}
    assert module_2.ATTRIBUTE_ENDS == {'(', '[', ']', ';', '@', '>', ' ', "'", '%', '\\', ')', '\t', '/', '*', ':', '?', '<', ',', '"', '='}
    assert module_2.EXTENDED_ATTRIBUTE_ENDS == {'(', '[', ']', ';', '@', '>', ' ', "'", '\\', ')', '\t', '/', '*', ':', '?', '<', ',', '"', '='}
    assert module_2.NLSET == {'\n', '\r'}
    assert module_2.SPECIALSNL == {'.', '<', ',', '(', '[', ']', '\n', '>', '\\', ')', '"', '\r', ';', '@', ':'}
    assert f'{type(module_2.rfc2047_matcher).__module__}.{type(module_2.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.DOT).__module__}.{type(module_2.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.DOT) == 1
    assert f'{type(module_2.ListSeparator).__module__}.{type(module_2.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.ListSeparator) == 1
    assert f'{type(module_2.RouteComponentMarker).__module__}.{type(module_2.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.RouteComponentMarker) == 1
    var_2 = module_0.identify_imports_main(var_1, var_1)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = '--color'
    var_4 = [var_3]
    var_5 = '-ndt!'
    var_6 = '  '
    var_7 = [var_5, var_6]
    var_8 = module_0.parse_args(var_7)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_9 = module_0.parse_args(var_2)
    var_10 = '--order-by-type'
    var_11 = [var_10]
    var_12 = module_0.parse_args(var_11)
    var_13 = '--dont-order-by-type'
    var_14 = [var_13]
    var_15 = module_0.parse_args(var_14)
    var_16 = '--multi-line'
    var_17 = '5'
    var_18 = [var_16, var_17]
    var_19 = module_0.parse_args(var_18)
    var_20 = module_0.parse_args(var_7)
    module_3.abspath(var_4)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = '-ndt!'
    var_4 = [var_3, var_3]
    var_5 = module_0.parse_args(var_4)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_6 = module_0.parse_args(var_2)
    var_7 = '--order-by-type'
    var_8 = [var_7]
    var_9 = module_0.parse_args(var_8)
    var_10 = [var_1]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--multi-line'
    var_13 = '5'
    var_14 = [var_12, var_13]
    var_15 = module_0.parse_args(var_14)
    var_16 = module_0.parse_args(var_4)
    module_3.abspath(var_14)

def test_case_9():
    var_0 = 'from os import path\nfrom sys import argv\nfrom collections import defaultdict'
    var_1 = 'import os\n'
    var_2 = [var_0]
    var_3 = module_0.identify_imports_main(var_2)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_4 = '-'
    var_5 = '--follow-links'
    var_6 = [var_4, var_5]
    var_7 = module_0.identify_imports_main(var_6, var_1)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '--line-length'
    var_1 = '88'
    var_2 = [var_0, var_1]
    var_3 = module_0.parse_args(var_2)
    assert module_0.ASCII_ART == "\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n"
    assert module_0.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_0.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_0.DEPRECATED_SINGLE_DASH_ARGS == {'-fgw', '-ds', '-fas', '-nlb', '-lbt', '-ws', '-ac', '-df', '-fss', '-dt', '-af', '-ot', '-nis', '-ca', '-tc', '-cs', '-sg', '-sl', '-wl', '-sd', '-rr', '-le', '-ff', '-fass', '-lai', '-sp', '-ls'}
    assert module_0.QUICK_GUIDE == "\n\n                 _                 _\n                (_) ___  ___  _ __| |_\n                | |/ _/ / _ \\/ '__  _/\n                | |\\__ \\/\\_\\/| |  | |_\n                |_|\\___/\\___/\\_/   \\_/\n\n      isort your imports, so you don't have to.\n\n                    VERSION 7.0.0\n\n\nNothing to do: no files or paths have been passed in!\n\nTry one of the following:\n\n    `isort .` - sort all Python files, starting from the current directory, recursively.\n    `isort . --interactive` - Do the same, but ask before making any changes.\n    `isort . --check --diff` - Check to see if imports are correctly sorted within this project.\n    `isort --help` - In-depth information about isort's available command-line options.\n\nVisit https://pycqa.github.io/isort/ for complete information about how to use isort.\n"
    var_4 = '--color'
    var_5 = [var_4]
    var_6 = module_0.parse_args(var_5)
    var_7 = module_0.parse_args(var_2)
    var_8 = '--indent'
    var_9 = '  '
    var_10 = [var_8, var_9]
    var_11 = module_0.parse_args(var_10)
    var_12 = '--treat-comment-as-code'
    var_13 = '# noqa'
    var_14 = '# type:'
    var_15 = [var_12, var_13, var_12, var_14]
    var_16 = module_0.parse_args(var_15)
    var_17 = '--order-by-type'
    var_18 = [var_17]
    var_19 = module_0.parse_args(var_18)
    var_20 = '--dont-order-by-type'
    var_21 = [var_20]
    var_22 = module_0.parse_args(var_21)
    var_23 = '--multi-line'
    var_24 = '5'
    var_25 = [var_23, var_24]
    var_26 = module_0.parse_args(var_25)
    var_27 = 'VERTICAL_GRID_GROUPED'
    var_28 = [var_23, var_27]
    var_29 = module_0.parse_args(var_28)
    module_3.abspath(var_5)