# Check out: https://github.com/GlowCheese/deepmosa
Error while converting AST module to output string: TypeError: sequence item 448: expected str instance, NoneType found
 Traceback (most recent call last):
  File "/workspace/pynguin/testcase/export.py", line 219, in save_module_to_file
    output = module_to_output_str(module, format_with_black=format_with_black)
  File "/workspace/pynguin/testcase/export.py", line 192, in module_to_output_str
    output = ast.unparse(ast.fix_missing_locations(module))
  File "/usr/local/lib/python3.10/ast.py", line 1681, in unparse
    return unparser.visit(ast_obj)
  File "/usr/local/lib/python3.10/ast.py", line 816, in visit
    return "".join(self._source)
TypeError: sequence item 448: expected str instance, NoneType found


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
          value=Constant(value='"key"}')),
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
          value=Constant(value=b'\xd9C\x13\x1d')),
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
        Assign(
          targets=[
            Name(id='var_0', ctx=Store())],
          value=Constant(value=None)),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
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
      name='test_case_3',
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
          value=Constant(value='{(&\x0c[=')),
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
          value=Constant(value=b'\xecf{o<')),
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
          value=Constant(value='4~){o ?vk;@?BoWOwBN')),
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
          value=Constant(value=b'\xf4\x9a')),
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
          value=Constant(value='[\n')),
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
          value=Constant(value='n_Yh)~wmhLO+",8NraA')),
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
          value=Constant(value='tEu|&=Re0~aUf~')),
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
          value=Constant(value='{(&\x0c')),
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
          value=Constant(value='{ ": ')),
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
          value=Constant(value='{}&\x0c')),
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
              Constant(value='re.Pattern')]))],
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
              Constant(value='re.Pattern')]))],
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
          value=Constant(value='{"key": }')),
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
          value=Constant(value='{"a&S": 1 "b": 2}')),
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
          value=Constant(value='{"ah 0 "b: 9}')),
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
          value=Constant(value='{"a":  "b": 2}')),
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
          value=Constant(value='{"":E>Z ')),
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
          value=Constant(value='{"": ')),
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
          value=Constant(value='{"a": 1, "b": 2}')),
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
          value=Constant(value='{"": 3')),
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
          value=Constant(value='{"ey": "')),
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
          value=Constant(value='{"ey"\n: ')),
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
          value=Constant(value='3.14')),
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
      name='test_case_26',
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
              Constant(value=None)])),
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
              Constant(value='null')])),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Attribute(
            value=Attribute(
              value=Name(id='var_1', ctx=Load()),
              attr='start',
              ctx=Load()),
            attr='line_no',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_4', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=1)])),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Attribute(
            value=Attribute(
              value=Name(id='var_1', ctx=Load()),
              attr='start',
              ctx=Load()),
            attr='column_no',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_5', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=1)])),
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
                Name(id='var_6', ctx=Store())],
              value=Attribute(
                value=Attribute(
                  value=Name(id='var_5', ctx=Load()),
                  attr='start',
                  ctx=Load()),
                attr='char_index',
                ctx=Load()))]),
        Assert(
          test=Compare(
            left=Name(id='var_6', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=0)]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_27',
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
          value=Constant(value='{"a": 1,` "b": 2}')),
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
      name='test_case_28',
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
          value=Constant(value='3.1e4')),
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
      decorator_list=[])],
  type_ignores=[])