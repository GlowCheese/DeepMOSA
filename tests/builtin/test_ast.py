import ast
import pytest
from grappa import should

from typing import cast
from collections.abc import Iterable
from ast import (
    arg, Name, Constant, Load,
    keyword, Starred, Attribute,
    List, Dict
)


def check_ast_eq(u, v):
    u | should.be.a(type(v))
    if isinstance(u, (str, bytes)):
        u | should.equal(v)
    elif isinstance(u, Iterable):
        u | should.have.length.of(len(v))
        for x, y in zip(u, v):
            check_ast_eq(x, y)
    elif isinstance(u, ast.AST):
        ast.dump(u) | should.equal(ast.dump(v))
    else:
        u | should.equal(v)


@pytest.mark.parametrize(
    "func, expected",
    [
        (
            'def f(x): pass',
            {
                'args': [arg('x')]
            }
        ),
        (
            'def f(*, y): pass',
            {
                'kwonlyargs': [arg('y')],
                'kw_defaults': [None]
            }
        ),
        (
            'def f(x, *, y): pass',
            {
                'args': [arg('x')],
                'kwonlyargs': [arg('y')],
                'kw_defaults': [None]
            }
        ),
        (
            'def f(x=z): pass',
            {
                'args': [arg('x')],
                'defaults': [Name('z', ctx=Load())]
            }
        ),
        (
            'def f(*, y=z): pass',
            {
                'kwonlyargs': [arg('y')],
                'kw_defaults': [Name('z', ctx=Load())]
            }
        ),
        (
            'def f(*args): pass',
            {
                'vararg': arg('args')
            }
        ),
        (
            'def f(*args, y): pass',
            {
                'vararg': arg('args'),
                'kwonlyargs': [arg('y')],
                'kw_defaults': [None]
            }
        ),
        (
            'def f(x, **kwargs): pass',
            {
                'args': [arg('x')],
                'kwarg': arg('kwargs')
            }
        ),
        (
            'def f(x, y=z): pass',
            {
                'args': [arg('x'), arg('y')],
                'defaults': [Name('z', ctx=Load())]
            }
        ),
        (
            'def f(a, b, /, c, *args, d, e=5, **kwargs): pass',
            {
                'posonlyargs': [arg('a'), arg('b')],
                'vararg': arg('args'),
                'kwonlyargs': [arg('d'), arg('e')],
                'kw_defaults': [None, Constant(5)],
                'kwarg': arg('kwargs')
            }
        )
    ]
)
def test_func_def(func: str, expected: dict):
    node = cast(ast.arguments, ast.parse(func).body[0].args)
    for k, v in expected.items():
        check_ast_eq(getattr(node, k), v)


@pytest.mark.parametrize(
    "call_str, expected",
    [
        (
            'f()',
            {
                'func': Name('f', ctx=Load()),
                'args': [],
                'keywords': []
            }
        ),
        (
            'obj.method()',
            {
                'func': Attribute(Name('obj', ctx=Load()), 'method', ctx=Load()),
                'args': [],
                'keywords': []
            }
        ),
        (
            'f(a, b, *c, d=4, **e)',
            {
                'func': Name('f', ctx=Load()),
                'args': [
                    Name('a', ctx=Load()),
                    Name('b', ctx=Load()),
                    Starred(Name('c', ctx=Load()), ctx=Load())
                ],
                'keywords': [
                    keyword('d', Constant(4)),
                    keyword(None, Name('e', ctx=Load()))
                ]
            }
        ),
        (
            'f(*[1, 2], **{a: 4, 5: 6})',
            {
                'func': Name('f', ctx=Load()),
                'args': [
                    Starred(
                        List([Constant(1), Constant(2)], ctx=Load()),
                        ctx=Load()
                    )
                ],
                'keywords': [
                    keyword(None, Dict(
                        [Name('a', ctx=Load()), Constant(5)],
                        [Constant(4), Constant(6)],
                        ctx=Load()
                    ))
                ]
            }
        ),
        (
            'f(**{})',
            {
                'func': Name('f', ctx=Load()),
                'keywords': [
                    keyword(None, Dict([], [], ctx=Load()))
                ]
            }
        ),
    ]
)
def test_call_args(call_str: str, expected: dict):
    call_node = cast(ast.Call, ast.parse(call_str, mode='eval').body)
    for key, expected_value in expected.items():
        actual_value = getattr(call_node, key)
        check_ast_eq(actual_value, expected_value)
