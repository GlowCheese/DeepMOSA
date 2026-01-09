# Check out: https://github.com/GlowCheese/deepmosa
Error while converting AST module to output string: TypeError: sequence item 234: expected str instance, NoneType found
 Traceback (most recent call last):
  File "/workspace/pynguin/testcase/export.py", line 219, in save_module_to_file
    output = module_to_output_str(module, format_with_black=format_with_black)
  File "/workspace/pynguin/testcase/export.py", line 192, in module_to_output_str
    output = ast.unparse(ast.fix_missing_locations(module))
  File "/usr/local/lib/python3.10/ast.py", line 1681, in unparse
    return unparser.visit(ast_obj)
  File "/usr/local/lib/python3.10/ast.py", line 816, in visit
    return "".join(self._source)
TypeError: sequence item 234: expected str instance, NoneType found


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
          value=Constant(value='t3|U*yn3')),
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
          value=Constant(value=b'R\x03\x90=]8K\xd5\x0b\xf7\xf9\xfb')),
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
      name='test_case_5',
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Constant(value=b'{"name": "John", "age": 30}')),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
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
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Constant(value='')),
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
                  Name(id='var_4', ctx=Load())],
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
          value=Constant(value=b'\xdffM\xac\xfeCW\x82\xe8\x86c\xf2A\x8c\xa0,')),
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
          value=Constant(value='nZ')),
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Constant(value='{"name": "John", "age": 30')),
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=BinOp(
            left=Name(id='var_2', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value='[1, 2, 3]')),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=BinOp(
            left=Name(id='var_7', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Constant(value='42')),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_10', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=BinOp(
            left=Name(id='var_11', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_13', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_14', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
            Name(id='var_15', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_16', ctx=Store())],
          value=BinOp(
            left=Name(id='var_15', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_17', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_18', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
            Name(id='var_19', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_20', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_21', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_20', ctx=Load())],
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
                        Name(id='var_21', ctx=Load())],
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
                        Name(id='var_21', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_22', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_20', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_23', ctx=Store())],
          value=BinOp(
            left=Name(id='var_22', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_24', ctx=Store())],
          value=Constant(value='[]')),
        Assign(
          targets=[
            Name(id='var_25', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_24', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_26', ctx=Store())],
          value=BinOp(
            left=Name(id='var_25', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_27', ctx=Store())],
          value=Constant(value='{"person": {"name": "Alice", "age": 25}}')),
        Assign(
          targets=[
            Name(id='var_28', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_27', ctx=Load())],
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
                        Name(id='var_28', ctx=Load())],
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
                        Name(id='var_28', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_29', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_27', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_30', ctx=Store())],
          value=BinOp(
            left=Name(id='var_29', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_31', ctx=Store())],
          value=Constant(value='[[1, 2], [3, 4]]')),
        Assign(
          targets=[
            Name(id='var_32', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_31', ctx=Load())],
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
                        Name(id='var_32', ctx=Load())],
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
                        Name(id='var_32', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_33', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_31', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_34', ctx=Store())],
          value=BinOp(
            left=Name(id='var_33', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_35', ctx=Store())],
          value=Constant(value='  { "key" : "value" }  ')),
        Assign(
          targets=[
            Name(id='var_36', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_35', ctx=Load())],
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
                        Name(id='var_36', ctx=Load())],
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
                        Name(id='var_36', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_37', ctx=Store())],
          value=Constant(value=3)),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Name(id='TypeError', ctx=Load())],
                keywords=[]))],
          body=[
            Assign(
              targets=[
                Name(id='var_38', ctx=Store())],
              value=BinOp(
                left=Name(id='var_31', ctx=Load()),
                op=Sub(),
                right=Name(id='var_37', ctx=Load())))])],
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
          value=Constant(value='{"name": John", "age" 30}')),
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
          value=Constant(value='{"name"+ John", "age" 30}')),
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=Constant(value='[1, 2, 3]')),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_4', ctx=Load())],
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
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_4', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=BinOp(
            left=Name(id='var_6', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=Constant(value='"Hello, World!"')),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_8', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=BinOp(
            left=Name(id='var_9', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Constant(value='42')),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_11', ctx=Load())],
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
                        Name(id='var_12', ctx=Load())],
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
                        Name(id='var_12', ctx=Load())],
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
            Name(id='var_13', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_11', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_14', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_15', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_14', ctx=Load())],
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
                        Name(id='var_15', ctx=Load())],
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
                        Name(id='var_15', ctx=Load())],
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
            Name(id='var_16', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_14', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_17', ctx=Store())],
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
                        Name(id='var_17', ctx=Load())],
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
                        Name(id='var_17', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_18', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_14', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_19', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_20', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_19', ctx=Load())],
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
                        Name(id='var_20', ctx=Load())],
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
                        Name(id='var_20', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_21', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_19', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_22', ctx=Store())],
          value=BinOp(
            left=Name(id='var_21', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_23', ctx=Store())],
          value=Constant(value='[]')),
        Assign(
          targets=[
            Name(id='var_24', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_23', ctx=Load())],
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
                        Name(id='var_24', ctx=Load())],
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
                        Name(id='var_24', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_25', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_23', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_26', ctx=Store())],
          value=BinOp(
            left=Name(id='var_25', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_27', ctx=Store())],
          value=Constant(value='{"person": {"name": "Alice", "age": 25}}')),
        Assign(
          targets=[
            Name(id='var_28', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_27', ctx=Load())],
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
                        Name(id='var_28', ctx=Load())],
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
                        Name(id='var_28', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_29', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_27', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_30', ctx=Store())],
          value=BinOp(
            left=Name(id='var_29', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_31', ctx=Store())],
          value=Constant(value='[[1, 2], [3, 4]]')),
        Assign(
          targets=[
            Name(id='var_32', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_31', ctx=Load())],
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
                        Name(id='var_32', ctx=Load())],
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
                        Name(id='var_32', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_33', ctx=Store())],
          value=BinOp(
            left=Name(id='var_6', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_34', ctx=Store())],
          value=Constant(value='  { "key" : "value" }  ')),
        Assign(
          targets=[
            Name(id='var_35', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_34', ctx=Load())],
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
                        Name(id='var_35', ctx=Load())],
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
                        Name(id='var_35', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_36', ctx=Store())],
          value=Constant(value=3)),
        Assign(
          targets=[
            Name(id='var_37', ctx=Store())],
          value=BinOp(
            left=Name(id='var_25', ctx=Load()),
            op=Sub(),
            right=Name(id='var_36', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_38', ctx=Store())],
          value=Constant(value='"Hello\\nWorld"')),
        Assign(
          targets=[
            Name(id='var_39', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_38', ctx=Load())],
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
                        Name(id='var_39', ctx=Load())],
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
                        Name(id='var_39', ctx=Load())],
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
            Name(id='var_40', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_38', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_41', ctx=Store())],
          value=BinOp(
            left=Name(id='var_40', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_42', ctx=Store())],
          value=Constant(value='"Hello A"')),
        Assign(
          targets=[
            Name(id='var_43', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_42', ctx=Load())],
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
                        Name(id='var_43', ctx=Load())],
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
                        Name(id='var_43', ctx=Load())],
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
            Name(id='var_44', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_42', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_45', ctx=Store())],
          value=BinOp(
            left=Name(id='var_44', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_46', ctx=Store())],
          value=Constant(value='1.23eM4')),
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
                  Name(id='var_46', ctx=Load())],
                keywords=[]))])],
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
          value=Constant(value='[1, 2,')),
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=BinOp(
            left=Name(id='var_2', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value='[1, 2, 3]')),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=BinOp(
            left=Name(id='var_7', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Constant(value='"Hello, World!"')),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_10', ctx=Load())],
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
                        Name(id='var_10', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_9', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=BinOp(
            left=Name(id='var_11', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_13', ctx=Store())],
          value=Constant(value='42')),
        Assign(
          targets=[
            Name(id='var_14', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
            Name(id='var_15', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_16', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_17', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_16', ctx=Load())],
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
                        Name(id='var_17', ctx=Load())],
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
                        Name(id='var_17', ctx=Load())],
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
            Name(id='var_18', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_16', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_19', ctx=Store())],
          value=BinOp(
            left=Name(id='var_18', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_20', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_21', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ListToken',
              ctx=Load()),
            args=[
              Name(id='var_2', ctx=Load()),
              Name(id='var_2', ctx=Load()),
              Name(id='var_11', ctx=Load())],
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
                        Name(id='var_21', ctx=Load())],
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
                        Name(id='var_21', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_22', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_20', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_23', ctx=Store())],
          value=BinOp(
            left=Name(id='var_22', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_24', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_25', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_24', ctx=Load())],
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
                        Name(id='var_25', ctx=Load())],
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
                        Name(id='var_25', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_26', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_24', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_27', ctx=Store())],
          value=BinOp(
            left=Name(id='var_15', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_28', ctx=Store())],
          value=Constant(value='[]')),
        Assign(
          targets=[
            Name(id='var_29', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_28', ctx=Load())],
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
                        Name(id='var_29', ctx=Load())],
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
                        Name(id='var_29', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_30', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_28', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_31', ctx=Store())],
          value=BinOp(
            left=Name(id='var_30', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_32', ctx=Store())],
          value=Constant(value='{"person": {"name": "Alice", "age": 25}}')),
        Assign(
          targets=[
            Name(id='var_33', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_32', ctx=Load())],
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
                        Name(id='var_33', ctx=Load())],
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
                        Name(id='var_33', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_34', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_32', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_35', ctx=Store())],
          value=BinOp(
            left=Name(id='var_34', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_36', ctx=Store())],
          value=Constant(value='[[1, 2], [3, 4]]')),
        Assign(
          targets=[
            Name(id='var_37', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_36', ctx=Load())],
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
                        Name(id='var_37', ctx=Load())],
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
                        Name(id='var_37', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_38', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_36', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_39', ctx=Store())],
          value=Constant(value='  { "key" : "value" }  ')),
        Assign(
          targets=[
            Name(id='var_40', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_39', ctx=Load())],
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
                        Name(id='var_40', ctx=Load())],
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
                        Name(id='var_40', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_41', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_39', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_42', ctx=Store())],
          value=Constant(value=3)),
        Assign(
          targets=[
            Name(id='var_43', ctx=Store())],
          value=BinOp(
            left=Name(id='var_41', ctx=Load()),
            op=Sub(),
            right=Name(id='var_42', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_44', ctx=Store())],
          value=Constant(value='"Hello\\nWorld"')),
        Assign(
          targets=[
            Name(id='var_45', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_44', ctx=Load())],
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
                        Name(id='var_45', ctx=Load())],
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
                        Name(id='var_45', ctx=Load())],
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
            Name(id='var_46', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_44', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_47', ctx=Store())],
          value=BinOp(
            left=Name(id='var_46', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_48', ctx=Store())],
          value=Constant(value='"Hello A"')),
        Assign(
          targets=[
            Name(id='var_49', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_48', ctx=Load())],
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
                        Name(id='var_49', ctx=Load())],
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
                        Name(id='var_49', ctx=Load())],
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
            Name(id='var_50', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_48', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_51', ctx=Store())],
          value=BinOp(
            left=Name(id='var_50', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_52', ctx=Store())],
          value=Constant(value='1.23e-4')),
        Assign(
          targets=[
            Name(id='var_53', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_52', ctx=Load())],
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
                        Name(id='var_53', ctx=Load())],
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
                        Name(id='var_53', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ScalarToken')])),
        With(
          items=[
            withitem(
              context_expr=Call(
                func=Attribute(
                  value=Name(id='pytest', ctx=Load()),
                  attr='raises',
                  ctx=Load()),
                args=[
                  Name(id='TypeError', ctx=Load())],
                keywords=[]))],
          body=[
            Assign(
              targets=[
                Name(id='var_54', ctx=Store())],
              value=BinOp(
                left=Name(id='var_32', ctx=Load()),
                op=Sub(),
                right=Name(id='var_1', ctx=Load())))])],
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
          value=Constant(value='{"name": "John", "age": 30, city": "New York"}')),
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=BinOp(
            left=Name(id='var_2', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value='[1, 2, 3]')),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=BinOp(
            left=Name(id='var_7', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Constant(value='42')),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_10', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=BinOp(
            left=Name(id='var_11', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_13', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_14', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
            Name(id='var_15', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_16', ctx=Store())],
          value=BinOp(
            left=Name(id='var_15', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_17', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_18', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
            Name(id='var_19', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_20', ctx=Store())],
          value=BinOp(
            left=Name(id='var_19', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_21', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_22', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_21', ctx=Load())],
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
                        Name(id='var_22', ctx=Load())],
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
                        Name(id='var_22', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_23', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_21', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_24', ctx=Store())],
          value=BinOp(
            left=Name(id='var_23', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_25', ctx=Store())],
          value=Constant(value='[]')),
        Assign(
          targets=[
            Name(id='var_26', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_25', ctx=Load())],
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
                        Name(id='var_26', ctx=Load())],
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
                        Name(id='var_26', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_27', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_25', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_28', ctx=Store())],
          value=BinOp(
            left=Name(id='var_27', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_29', ctx=Store())],
          value=Constant(value='{"person": {"name": "Alice", "age": 25}}')),
        Assign(
          targets=[
            Name(id='var_30', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_29', ctx=Load())],
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
                        Name(id='var_30', ctx=Load())],
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
                        Name(id='var_30', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_31', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_29', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_32', ctx=Store())],
          value=BinOp(
            left=Name(id='var_31', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_33', ctx=Store())],
          value=Constant(value='[[1, 2], [3, 4]]')),
        Assign(
          targets=[
            Name(id='var_34', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_33', ctx=Load())],
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
                        Name(id='var_34', ctx=Load())],
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
                        Name(id='var_34', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_35', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_33', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_36', ctx=Store())],
          value=BinOp(
            left=Name(id='var_35', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_37', ctx=Store())],
          value=Constant(value='  { "key" : "value" }  ')),
        Assign(
          targets=[
            Name(id='var_38', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_37', ctx=Load())],
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
                        Name(id='var_38', ctx=Load())],
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
                        Name(id='var_38', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_39', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_37', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_40', ctx=Store())],
          value=Constant(value=3)),
        Assign(
          targets=[
            Name(id='var_41', ctx=Store())],
          value=BinOp(
            left=Name(id='var_39', ctx=Load()),
            op=Sub(),
            right=Name(id='var_40', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_42', ctx=Store())],
          value=Constant(value='{]tmq&')),
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
                  Name(id='var_42', ctx=Load())],
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
          value=Constant(value='{"name": "John", "age": 30}')),
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
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_0', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_3', ctx=Store())],
          value=Constant(value=1)),
        Assign(
          targets=[
            Name(id='var_4', ctx=Store())],
          value=BinOp(
            left=Name(id='var_2', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_5', ctx=Store())],
          value=Constant(value='[1, 2, 3]')),
        Assign(
          targets=[
            Name(id='var_6', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
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
                        Name(id='var_6', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_7', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_8', ctx=Store())],
          value=BinOp(
            left=Name(id='var_7', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_9', ctx=Store())],
          value=Constant(value='"Hello, World!"')),
        Assign(
          targets=[
            Name(id='var_10', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_5', ctx=Load())],
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
                        Name(id='var_10', ctx=Load())],
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
                        Name(id='var_10', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_11', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_9', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_12', ctx=Store())],
          value=BinOp(
            left=Name(id='var_11', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_13', ctx=Store())],
          value=Constant(value='42')),
        Assign(
          targets=[
            Name(id='var_14', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
                        Name(id='var_14', ctx=Load())],
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
            Name(id='var_15', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_13', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_16', ctx=Store())],
          value=BinOp(
            left=Name(id='var_15', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_17', ctx=Store())],
          value=Constant(value='true')),
        Assign(
          targets=[
            Name(id='var_18', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
                        Name(id='var_18', ctx=Load())],
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
            Name(id='var_19', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_17', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_20', ctx=Store())],
          value=BinOp(
            left=Name(id='var_19', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_21', ctx=Store())],
          value=Constant(value='null')),
        Assign(
          targets=[
            Name(id='var_22', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_3', ctx=Load()),
              attr='ListToken',
              ctx=Load()),
            args=[
              Name(id='var_2', ctx=Load()),
              Name(id='var_2', ctx=Load()),
              Name(id='var_16', ctx=Load())],
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
                        Name(id='var_22', ctx=Load())],
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
                        Name(id='var_22', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_23', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_21', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_24', ctx=Store())],
          value=BinOp(
            left=Name(id='var_23', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_25', ctx=Store())],
          value=Constant(value='{}')),
        Assign(
          targets=[
            Name(id='var_26', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_25', ctx=Load())],
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
                        Name(id='var_26', ctx=Load())],
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
                        Name(id='var_26', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.DictToken')])),
        Assign(
          targets=[
            Name(id='var_27', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_25', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_28', ctx=Store())],
          value=BinOp(
            left=Name(id='var_15', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_29', ctx=Store())],
          value=Constant(value='[]')),
        Assign(
          targets=[
            Name(id='var_30', ctx=Store())],
          value=Call(
            func=Attribute(
              value=Name(id='module_0', ctx=Load()),
              attr='tokenize_json',
              ctx=Load()),
            args=[
              Name(id='var_29', ctx=Load())],
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
                        Name(id='var_30', ctx=Load())],
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
                        Name(id='var_30', ctx=Load())],
                      keywords=[]),
                    attr='__qualname__',
                    ctx=Load()),
                  conversion=-1)]),
            ops=[
              Eq()],
            comparators=[
              Constant(value='typesystem.tokenize.tokens.ListToken')])),
        Assign(
          targets=[
            Name(id='var_31', ctx=Store())],
          value=Call(
            func=Name(id='len', ctx=Load()),
            args=[
              Name(id='var_29', ctx=Load())],
            keywords=[])),
        Assign(
          targets=[
            Name(id='var_32', ctx=Store())],
          value=BinOp(
            left=Name(id='var_31', ctx=Load()),
            op=Sub(),
            right=Name(id='var_3', ctx=Load()))),
        Assign(
          targets=[
            Name(id='var_33', ctx=Store())],
          value=Constant(value='{"perOon": {"name": "Alice<, "age": 25}}')),
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
                  Name(id='var_33', ctx=Load())],
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
          value=Constant(value='{"name": "John", "age":30')),
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