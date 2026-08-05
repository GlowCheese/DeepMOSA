# Check out: https://github.com/GlowCheese/deepmosa
Error while converting AST module to output string: TypeError: sequence item 241: expected str instance, NoneType found
 Traceback (most recent call last):
  File "/workspace/pynguin/testcase/export.py", line 219, in save_module_to_file
    output = module_to_output_str(module, format_with_black=format_with_black)
  File "/workspace/pynguin/testcase/export.py", line 192, in module_to_output_str
    output = ast.unparse(ast.fix_missing_locations(module))
  File "/usr/local/lib/python3.10/ast.py", line 1681, in unparse
    return unparser.visit(ast_obj)
  File "/usr/local/lib/python3.10/ast.py", line 816, in visit
    return "".join(self._source)
TypeError: sequence item 241: expected str instance, NoneType found


Formatted AST dump of the module:
Module(
  body=[
    Import(
      names=[
        alias(name='pytest')]),
    Import(
      names=[
        alias(name='typesystem.tokenize.tokenize_json', asname='module_0')]),
    Import(
      names=[
        alias(name='typesystem.base', asname='module_1')]),
    Import(
      names=[
        alias(name='re', asname='module_2')]),
    Import(
      names=[
        alias(name='typesystem.tokenize.tokens', asname='module_3')]),
    FunctionDef(
      name='test_case_0',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='T$Powmp^|zOd-\x0b.')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_1',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value=b'q\x12\xb1@\xb8\xd3?6\x02R\xe7%?')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_2',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Pass()],
      decorator_list=[]),
    FunctionDef(
      name='test_case_3',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='_TokenizingDecoder',
              ctx=Load()),
            args=[],
            keywords=[]))],
      decorator_list=[
        Call(
          func=Attribute(
            value=Attribute(
              value=Name(id='pytest', ctx=Load()),
              attr='mark',
              ctx=Load()),
            attr='xfail',
            ctx=Load()),
          args=[],
          keywords=[
            keyword(
              arg='strict',
              value=Constant(value=True))])]),
    FunctionDef(
      name='test_case_4',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value="cd'yk4Wj\tSW_8.4r5\n]")),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='validate_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load()),
              Name(id='var_0', ctx=Load())],
            keywords=[]))],
      decorator_list=[
        Call(
          func=Attribute(
            value=Attribute(
              value=Name(id='pytest', ctx=Load()),
              attr='mark',
              ctx=Load()),
            attr='xfail',
            ctx=Load()),
          args=[],
          keywords=[
            keyword(
              arg='strict',
              value=Constant(value=True))])]),
    FunctionDef(
      name='test_case_5',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='   ')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_6',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='1R[\\ZTd(u\x0cXMi :L0Z')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_7',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_8',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value=0)),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Subscript(
            value=Name(id='var_0', ctx=Load()),
            slice=Name(id='var_1', ctx=Load()),
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_2', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='n')])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value='n')),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Compare(
            left=Name(id='var_2', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Name(id='var_3', ctx=Load())])),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value=4)),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=BinOp(
            left=Name(id='var_1', ctx=Load()),
            op=Add(),
            right=Name(id='var_5', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Subscript(
            value=Name(id='var_0', ctx=Load()),
            slice=Slice(
              lower=Name(id='var_1', ctx=Load()),
              upper=Name(id='var_6', ctx=Load())),
            ctx=Load())),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_2', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_9',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='"D.Uu}:M_V*98I:')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_10',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='[1, "two"]')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='value',
            ctx=Load())),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='string',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_3', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='[1, "two"]')]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_11',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value="fNBu\\lc~a,'==Nu6)\r")),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_12',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n  "a": 1\n}')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_13',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{"key": "value"}')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='string',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_2', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='{"key": "value"}')]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_14',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='value',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_2', ctx=Load()),
            ops=[
              Is()],
            comparators=[
              Constant(value=True)]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_15',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='123.45')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='value',
            ctx=Load()))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_16',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='false')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_1', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Name(id='AttributeError', ctx=Load())],
                keywords=[]))],
          body=[
            Assign(
              targets=[
                Name(id='var_2', ctx=Store())],
              value=Attribute(
                value=Name(id='var_0', ctx=Load()),
                attr='value',
                ctx=Load()))]),
        Assert(
          test=Compare(
            left=Name(id='var_2', ctx=Load()),
            ops=[
              Is()],
            comparators=[
              Constant(value=False)]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_17',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='t[uz')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_18',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{key: "value"}')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value='\\s*')),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='compile',
              ctx=Load()),
            args=[
              Name(id='var_1', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_2', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_2', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='ASCII',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='A',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='IGNORECASE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='I',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='LOCALE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='L',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='UNICODE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='U',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='MULTILINE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='M',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DOTALL',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='S',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='VERBOSE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='X',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='TEMPLATE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='T',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DEBUG',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DEBUG',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.getset_descriptor')])),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='validate_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load()),
              Name(id='var_2', ctx=Load())],
            keywords=[]))],
      decorator_list=[
        Call(
          func=Attribute(
            value=Attribute(
              value=Name(id='pytest', ctx=Load()),
              attr='mark',
              ctx=Load()),
            attr='xfail',
            ctx=Load()),
          args=[],
          keywords=[
            keyword(
              arg='strict',
              value=Constant(value=True))])]),
    FunctionDef(
      name='test_case_19',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n  a": 1}')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_20',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Constant(value='\\s*')),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='compile',
              ctx=Load()),
            args=[
              Name(id='var_2', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_3', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_3', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='ASCII',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='A',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='IGNORECASE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='I',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='LOCALE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='L',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='UNICODE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='U',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='MULTILINE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='M',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DOTALL',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='S',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='VERBOSE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='X',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='TEMPLATE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='T',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DEBUG',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DEBUG',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.getset_descriptor')])),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Constant(value='val')),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value=2)),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Constant(value=4)),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ScalarToken',
              ctx=Load()),
            args=[
              Name(id='var_4', ctx=Load()),
              Name(id='var_5', ctx=Load()),
              Name(id='var_6', ctx=Load()),
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_7', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_7', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=Constant(value=5)),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_1', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_9', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_9', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Tuple(
            elts=[
              Name(id='var_7', ctx=Load()),
              Name(id='var_8', ctx=Load())],
            ctx=Load())),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Lambda(
            args=arguments(
              posonlyargs=[],
              args=[
                arg(arg='s'),
                arg(arg='e')],
              kwonlyargs=[],
              kw_defaults=[],
              defaults=[]),
            body=Name(id='var_10', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=Attribute(
            value=Name(id='var_3', ctx=Load()),
            attr='match',
            ctx=Load()))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_21',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n  "a":B 1\n}')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_22',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n  "a": 1\nA')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_23',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n " "a":1\nA')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_24',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{\n  "a": \nA')),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Attribute(
                    value=Name(id='module_1', ctx=Load()),
                    attr='ParseError',
                    ctx=Load())],
                keywords=[]))],
          body=[
            Expr(
              value=Call(
                func=Attribute(
                  value=Name(id='module_0', ctx=Load()),
                  attr='tokenize_json',
                  ctx=Load()),
                args=[
                  Name(id='var_0', ctx=Load())],
                keywords=[]))])],
      decorator_list=[]),
    FunctionDef(
      name='test_case_25',
      args=arguments(
        posonlyargs=[],
        args=[],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[]),
      body=[
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value='{"key": "value", "next": "value"}')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value='\\s*')),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_2', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_2', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='FLAGS',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr=None,
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='WHITESPACE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='WHITESPACE_STR',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=' \t\n\r')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Name(id='module_0', ctx=Load()),
                          attr='NUMBER_RE',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='compile',
              ctx=Load()),
            args=[
              Name(id='var_1', ctx=Load())],
            keywords=[])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_3', ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Name(id='var_3', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='re.Pattern')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='ASCII',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='A',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='ASCII',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='IGNORECASE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='I',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='IGNORECASE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='LOCALE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='L',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='LOCALE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='UNICODE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='U',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='UNICODE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='MULTILINE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='M',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='MULTILINE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DOTALL',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='S',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DOTALL',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='VERBOSE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='X',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='VERBOSE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='TEMPLATE',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='T',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='TEMPLATE',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='module_2', ctx=Load()),
              attr='DEBUG',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Attribute(
                value=Attribute(
                  value=Name(id='module_2', ctx=Load()),
                  attr='RegexFlag',
                  ctx=Load()),
                attr='DEBUG',
                ctx=Load())])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='pattern',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='flags',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groups',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.member_descriptor')])),
        Assert(
          test=Compare(
            left=JoinedStr(
              values=[
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__module__',
                    ctx=Load()),
                  conversion=-1),
                Constant(value='.'),
                FormattedValue(
                  value=Attribute(
                    value=Call(
                      func=Name(id='type', ctx=Load()),
                      args=[
                        Attribute(
                          value=Attribute(
                            value=Name(id='module_2', ctx=Load()),
                            attr='Pattern',
                            ctx=Load()),
                          attr='groupindex',
                          ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='builtins.getset_descriptor')]))],
      decorator_list=[])],
  type_ignores=[])