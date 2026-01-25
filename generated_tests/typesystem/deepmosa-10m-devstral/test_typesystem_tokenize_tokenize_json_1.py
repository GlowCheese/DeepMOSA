# Check out: https://github.com/GlowCheese/deepmosa
Error while converting AST module to output string: TypeError: sequence item 455: expected str instance, NoneType found
 Traceback (most recent call last):
  File "/workspace/pynguin/testcase/export.py", line 219, in save_module_to_file
    output = module_to_output_str(module, format_with_black=format_with_black)
  File "/workspace/pynguin/testcase/export.py", line 192, in module_to_output_str
    output = ast.unparse(ast.fix_missing_locations(module))
  File "/usr/local/lib/python3.10/ast.py", line 1681, in unparse
    return unparser.visit(ast_obj)
  File "/usr/local/lib/python3.10/ast.py", line 816, in visit
    return "".join(self._source)
TypeError: sequence item 455: expected str instance, NoneType found


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
          value=Constant(value='Dhe%')),
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
          value=Constant(value=b'\xc3\xce\xe71M\x06\xe9')),
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
          value=Constant(value='tTF')),
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
          value=Constant(value='9q3\t"Hs4B*e%3')),
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
          value=Constant(value='{"key1": "value1",  key2|: "value2"}')),
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
          value=Constant(value='fmr$X')),
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
          value=Constant(value=b'')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value=None)),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='validate_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load()),
              Name(id='var_1', ctx=Load())],
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
          value=Constant(value='{mJr^G\t')),
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
          value=Constant(value='[G~{=BNi\t|* hG7')),
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
          value=Constant(value='n}%L9 ;^+v')),
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
          value=Constant(value='{ J`GX')),
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
          value=Constant(value=b'[')),
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
          value=Constant(value='{}')),
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
          value=List(
            elts=[
              Name(id='var_0', ctx=Load()),
              Name(id='var_1', ctx=Load()),
              Name(id='var_0', ctx=Load())],
            ctx=Load())),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='var_1', ctx=Load()),
              attr='lookup_key',
              ctx=Load()),
            args=[
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
          value=Constant(value='{"*key": "vBalue"')),
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
          value=Constant(value='{"key1": "value1", "key2": "value2"}')),
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
          value=Constant(value='{"key1": "value1", "key2"0 "value2"}')),
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
          value=Constant(value='{"*ky1:"vBalu"')),
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
          value=Constant(value='{"*Tey": "vBalue')),
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
          value=Constant(value='{"*key":f "vBalue"')),
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
          value=Constant(value='{"key1": "value1" "key2"0 "value2"}')),
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
          value=Constant(value='7e6CF')),
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
          value=Constant(value='{"key" : "value"}')),
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
          value=Constant(value=7)),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Subscript(
            value=Name(id='var_0', ctx=Load()),
            slice=Slice(
              lower=Name(id='var_2', ctx=Load()),
              upper=BinOp(
                left=Name(id='var_2', ctx=Load()),
                op=Add(),
                right=Constant(value=1))),
            ctx=Load())),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='_TokenizingDecoder',
              ctx=Load()),
            args=[],
            keywords=[
              keyword(
                value=Name(id='var_3', ctx=Load()))]))],
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
          value=Constant(value='false')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value=None)),
        Expr(
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='validate_json',
              ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load()),
              Name(id='var_1', ctx=Load())],
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
              Constant(value='re.Pattern')])),
        Assign(
          targets=[
            Name(id='var_2', ctx=Store())],
          value=Constant(value=3.14)),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=0)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Constant(value=3)),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ScalarToken',
              ctx=Load()),
            args=[
              Name(id='var_2', ctx=Load()),
              Name(id='var_3', ctx=Load()),
              Name(id='var_4', ctx=Load()),
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
                        Name(id='var_5', ctx=Load())],
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
                        Name(id='var_5', ctx=Load())],
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
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Name(id='bool', ctx=Load()),
            args=[
              Compare(
                left=Name(id='var_1', ctx=Load()),
                ops=[
                  Eq()],
                comparators=[
                  Name(id='var_5', ctx=Load())])],
            keywords=[])),
        Assert(
          test=Compare(
            left=Name(id='var_6', ctx=Load()),
            ops=[
              Is()],
            comparators=[
              Constant(value=True)])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='value',
            ctx=Load())),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=Call(
            func=Name(id='bool', ctx=Load()),
            args=[
              Compare(
                left=Attribute(
                  value=Name(id='var_1', ctx=Load()),
                  attr='value',
                  ctx=Load()),
                ops=[
                  Eq()],
                comparators=[
                  Constant(value=3.14)])],
            keywords=[])),
        Assert(
          test=Compare(
            left=Name(id='var_8', ctx=Load()),
            ops=[
              Is()],
            comparators=[
              Constant(value=True)])),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='string',
            ctx=Load())),
        Assert(
          test=Compare(
            left=Name(id='var_9', ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value='3.14')])),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_1', ctx=Load()),
              attr='Position',
              ctx=Load()),
            args=[
              Name(id='var_10', ctx=Load()),
              Name(id='var_10', ctx=Load()),
              Name(id='var_3', ctx=Load())],
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
                        Name(id='var_11', ctx=Load())],
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
                        Name(id='var_11', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.base.Position')])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='var_11', ctx=Load()),
              attr='line_no',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=1)])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='var_11', ctx=Load()),
              attr='column_no',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=1)])),
        Assert(
          test=Compare(
            left=Attribute(
              value=Name(id='var_11', ctx=Load()),
              attr='char_index',
              ctx=Load()),
            ops=[
              Eq()],
            comparators=[
              Constant(value=0)])),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=Attribute(
            value=Name(id='var_1', ctx=Load()),
            attr='start',
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
                  Name(id='AttributeError', ctx=Load())],
                keywords=[]))],
          body=[
            Assign(
              targets=[
                Name(id='var_13', ctx=Store())],
              value=Call(
                func=Name(id='bool', ctx=Load()),
                args=[
                  Compare(
                    left=Attribute(
                      value=Name(id='var_11', ctx=Load()),
                      attr='start',
                      ctx=Load()),
                    ops=[
                      Eq()],
                    comparators=[
                      Name(id='var_11', ctx=Load())])],
                keywords=[]))]),
        Assert(
          test=Compare(
            left=Name(id='var_13', ctx=Load()),
            ops=[
              Is()],
            comparators=[
              Constant(value=True)]))],
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
          value=Constant(value='{ "a" : 1 }')),
        Assign(
          targets=[
            Name(id='var_1', ctx=Store())],
          value=Constant(value=0)),
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
          value=Constant(value=True)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ScalarToken',
              ctx=Load()),
            args=[
              Name(id='var_3', ctx=Load()),
              Name(id='var_1', ctx=Load()),
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
                        Name(id='var_4', ctx=Load())],
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
                        Name(id='var_4', ctx=Load())],
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
            Name(id='var_5', ctx=Store())],
          value=Tuple(
            elts=[
              Name(id='var_4', ctx=Load()),
              Name(id='var_3', ctx=Load())],
            ctx=Load())),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Lambda(
            args=arguments(
              posonlyargs=[],
              args=[
                arg(arg='s'),
                arg(arg='e')],
              kwonlyargs=[],
              kw_defaults=[],
              defaults=[]),
            body=Name(id='var_5', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ScalarToken',
              ctx=Load()),
            args=[
              Name(id='var_3', ctx=Load()),
              Name(id='var_1', ctx=Load()),
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
              Constant(value='typesystem.tokenize.tokens.ScalarToken')]))],
      decorator_list=[]),
    FunctionDef(
      name='test_case_29',
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
          value=Constant(value='{"key":\n "{?vBalue"')),
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
      decorator_list=[])],
  type_ignores=[])