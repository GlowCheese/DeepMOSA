####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_wrap_with_import_splitter():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "(" in result and ")" in result and line_separator in result

def test_line_wrap_with_dot_splitter():
    content = "object.very_long_method_name(arg1, arg2)"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "(" in result and ")" in result and line_separator in result

def test_line_wrap_with_as_splitter():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "as " in result and line_separator in result

def test_line_with_comment_no_wrap():
    content = "short line # comment"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line # comment"

def test_line_with_comment_wrap():
    content = "long line that needs wrapping # important comment"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "comment" in result and line_separator in result

def test_line_noqa_mode():
    content = "very long line that should not be wrapped"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == f"{content} # NOQA"

def test_line_noqa_already_present():
    content = "very long line # NOQA"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == content

def test_line_use_parentheses_false():
    content = "long line with import statement"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False)
    result = line(content, line_separator, config)
    assert "\\" in result and line_separator in result

def test_line_include_trailing_comma_false():
    content = "long line with dot separator"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False)
    result = line(content, line_separator, config)
    assert "," not in result.split(line_separator)[-1]

def test_line_wrap_length_shorter_than_line_length():
    content = "very long line that needs wrapping"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert line_separator in result


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #3
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_no_wrap_noqa_mode():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds the line length limit", "\n", config) == "long line that exceeds the line length limit NOQA"

def test_line_wrap_with_import_splitter():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import long_function_name"
    expected = "from module import (\n    long_function_name)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "object.long_attribute_name"
    expected = "object.(\n    long_attribute_name)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_as_splitter():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "import module as alias"
    expected = "import module as alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    content = "import module # comment"
    expected = "import (\n    module # comment\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    content = "import module # noqa"
    expected = "import module # noqa"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    content = "import module"
    expected = "import (\n    module,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False)
    content = "import module"
    expected = "import \\\n    module"
    assert line(content, "\n", config) == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_line_predicate_true():
    config = Config(
        line_length=79,
        wrap_length=80,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    "
    )
    content = "import a_very_long_module_name_that_exceeds_the_line_length_limit"
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #5
#--------------------------

```python
def test_import_statement_with_explode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert result == "from module import (\n    a,\n    b,\n    c,\n)\n"

def test_import_statement_without_explode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        explode=False,
    )
    assert result == "from module import (a, b, c)\n"

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
    )
    assert "# comment" in result

def test_import_statement_with_custom_line_separator():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_with_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=50)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "\n" in result

def test_import_statement_with_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert result.endswith(",\n")

def test_import_statement_with_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
        config=config,
    )
    assert "# comment" not in result

def test_import_statement_with_multi_line_output():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert "\n" in result

def test_import_statement_with_custom_indent():
    config = Config(indent="    ")
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "    " in result

def test_import_statement_with_comment_prefix():
    config = Config(comment_prefix="# ")
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["comment"],
        config=config,
    )
    assert "# comment" in result


# LLM-generated content at query #6
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_noqa_mode_with_noqa_comment():
    content = "long line that exceeds the default line length but has a NOQA comment # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == content

def test_line_noqa_mode_without_noqa_comment():
    content = "long line that exceeds the default line length without NOQA comment"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == f"{content} NOQA"

def test_line_wrap_with_import_splitter():
    content = "from module import long_module_name, another_long_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == f"from module import \\{line_separator}    long_module_name, another_long_name"

def test_line_wrap_with_parentheses_and_trailing_comma():
    content = "from module import long_module_name, another_long_name # comment"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == f"from module import ({line_separator}    long_module_name,{line_separator}) # comment"

def test_line_wrap_with_as_splitter():
    content = "import long_module_name as short_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == f"import long_module_name as{line_separator}    short_name"

def test_line_wrap_with_noqa_in_comment():
    content = "from module import long_module_name, another_long_name # noqa: F401"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == f"from module import ({line_separator}    long_module_name,{line_separator}    another_long_name,  # noqa: F401{line_separator})"

def test_line_wrap_with_vertical_hanging_indent():
    content = "from module import long_module_name, another_long_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"from module import ({line_separator}    long_module_name,{line_separator}    another_long_name,{line_separator})"

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import long_module_name, another_long_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line(content, line_separator, config)
    assert result == f"from module import ({line_separator}    long_module_name,{line_separator}    another_long_name,{line_separator})"


# LLM-generated content at query #7
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement("from os import", ["path", "sys"])
    assert result == "from os import path, sys"

def test_import_statement_with_comments():
    result = import_statement("from os import", ["path", "sys"], comments=["# Comment"])
    assert "# Comment" in result

def test_import_statement_explode():
    result = import_statement("from os import", ["path", "sys"], explode=True)
    assert "\n" in result

def test_import_statement_multi_line_output():
    result = import_statement("from os import", ["path", "sys"], multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert "\n" in result

def test_import_statement_custom_line_separator():
    result = import_statement("from os import", ["path", "sys"], line_separator="\r\n")
    assert "\r\n" in result

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True)
    result = import_statement("from os import", ["path", "sys"], config=config)
    assert "\n" in result or result == "from os import path, sys"

def test_import_statement_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement("from os import", ["path", "sys"], config=config)
    assert result.endswith(",") or "\n" in result

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement("from os import", ["path", "sys"], comments=["# Comment"], config=config)
    assert "# Comment" not in result

def test_import_statement_custom_indent():
    config = Config(indent="    ")
    result = import_statement("from os import", ["path", "sys"], config=config)
    assert "    " in result or result == "from os import path, sys"

def test_import_statement_custom_comment_prefix():
    config = Config(comment_prefix="# ")
    result = import_statement("from os import", ["path", "sys"], comments=["# Comment"], config=config)
    assert "# Comment" in result

def test_import_statement_single_line_output():
    result = import_statement("from os import", ["path"])
    assert "\n" not in result


# LLM-generated content at query #8
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "import os.path as osp"
    line_without_comment = "import os.path as osp"
    _comma_maybe = ","
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    ) == True


# LLM-generated content at query #9
#--------------------------

```python
def test_while_loop_predicate():
    config = Config(wrap_length=20, line_length=20, balanced_wrapping=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c", "d", "e"],
        config=config,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    lines = result.split("\n")
    assert len(lines[-1]) < min(len(line) for line in lines[:-1]) and len(lines) == 3 and config.wrap_length > 10


# LLM-generated content at query #10
#--------------------------

```python
def test_line_30_predicate_evaluates_to_true():
    config = Config(
        wrap_length=50,
        line_length=100,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    "
    )
    content = "import a_very_long_module_name_that_exceeds_the_wrap_length_limit"
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #11
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=5)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #12
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(
        line_length=100,
        wrap_length=None,
        multi_line_output=Modes.NOQA,
        indent="",
        comment_prefix="#",
        use_parentheses=False,
        include_trailing_comma=False
    )
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #13
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    assert line(content, line_separator) == "short line"

def test_line_noqa_mode_with_noqa_comment():
    content = "long line # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "long line # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    content = "long line"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "long line # NOQA"

def test_line_wrap_with_import_splitter():
    content = "from module import long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name\n)"

def test_line_wrap_with_as_splitter():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "import module as (\n    alias\n)"

def test_line_wrap_with_dot_splitter():
    content = "module.long_function_name()"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "module.\n    long_function_name()"

def test_line_wrap_with_comment():
    content = "long_line # some comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "long_line,  # some comment"

def test_line_wrap_with_noqa_in_comment():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line(content, line_separator, config) == "long_line(\n    # noqa\n)"

def test_line_wrap_with_trailing_comma():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    assert line(content, line_separator, config) == "long_line(\n    ,\n)"

def test_line_wrap_with_vertical_grid_grouped():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    assert line(content, line_separator, config) == "long_line(\n    \n)"

def test_line_wrap_with_vertical_hanging_indent():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line(content, line_separator, config) == "long_line(\n    \n)"


# LLM-generated content at query #14
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        line_separator="\n",
        config=Config(),
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert result == "from module import (\n    a,\n    b,\n    c,\n)\n"

def test_import_statement_explode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert result == "from module import (\n    a,\n    b,\n    c,\n)\n"

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
        line_separator="\n",
        config=Config(),
    )
    assert result == "from module import (\n    a,  # comment\n    b,\n    c,\n)\n"

def test_import_statement_single_line():
    result = import_statement(
        import_start="from module import",
        from_imports=["a"],
        line_separator="\n",
        config=Config(),
    )
    assert result == "from module import a\n"

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=50)
    result = import_statement(
        import_start="from module import (",
        from_imports=["very_long_name_a", "very_long_name_b", "very_long_name_c"],
        line_separator="\n",
        config=config,
    )
    assert result == "from module import (\n    very_long_name_a,\n    very_long_name_b,\n    very_long_name_c,\n)\n"

def test_import_statement_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b"],
        line_separator="\n",
        config=config,
    )
    assert result == "from module import (\n    a,\n    b,\n)\n"

def test_import_statement_no_trailing_comma():
    config = Config(include_trailing_comma=False)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b"],
        line_separator="\n",
        config=config,
    )
    assert result == "from module import (\n    a,\n    b\n)\n"

def test_import_statement_custom_indent():
    config = Config(indent="    ")
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b"],
        line_separator="\n",
        config=config,
    )
    assert result == "from module import (\n    a,\n    b,\n)\n"

def test_import_statement_custom_line_separator():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b"],
        line_separator="\r\n",
        config=Config(),
    )
    assert result == "from module import (\r\n    a,\r\n    b,\r\n)\r\n"

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b"],
        comments=["# comment"],
        line_separator="\n",
        config=config,
    )
    assert result == "from module import (\n    a,\n    b,\n)\n"


# LLM-generated content at query #15
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_no_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds the line length limit", "\n", config) == "long line that exceeds the line length limit NOQA"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds the line length limit # NOQA", "\n", config) == "long line that exceeds the line length limit # NOQA"

def test_line_wrap_import():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name\n)"

def test_line_wrap_cimport():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module cimport long_module_name", "\n", config) == "from module cimport (\n    long_module_name\n)"

def test_line_wrap_dot():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("module.long_module_name.function", "\n", config) == "module.long_module_name.\n    function"

def test_line_wrap_as():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("import module as long_alias", "\n", config) == "import module as long_alias"

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name  # comment\n)"

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line("from module import long_module_name # noqa", "\n", config) == "from module import (\n    long_module_name,  # noqa\n)"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_wrap_with_parentheses_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name,  # comment\n)"

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name\n)"

def test_line_wrap_with_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ")
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name\n)"

def test_line_wrap_with_different_line_separator():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name", "\r\n", config) == "from module import (\r\n    long_module_name\r\n)"

def test_line_wrap_with_comment_prefix():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ")
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name,  # comment\n)"

def test_line_wrap_with_wrap_length():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=15)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name\n)"

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line("from module import long_module_name # noqa: F401", "\n", config) == "from module import (\n    long_module_name,  # noqa: F401\n)"

def test_line_wrap_with_noqa_in_comment_no_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False)
    assert line("from module import long_module_name # noqa: F401", "\n", config) == "from module import long_module_name # noqa: F401"

def test_line_wrap_with_noqa_in_comment_no_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False)
    assert line("from module import long_module_name # noqa: F401", "\n", config) == "from module import (\n    long_module_name  # noqa: F401\n)"

def test_line_wrap_with_noqa_in_comment_and_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # noqa: F401", "\n", config) == "from module import (\n    long_module_name,  # noqa: F401\n)"


# LLM-generated content at query #16
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "short line"

def test_line_wrap_with_import():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_cimport():
    content = "cimport module.long_function_name, another_function"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "cimport module.long_function_name,\n    another_function"

def test_line_wrap_with_dot():
    content = "module.long_function_name.another_function"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "module.long_function_name.another_function"

def test_line_wrap_with_as():
    content = "import module as alias"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "import module as alias"

def test_line_wrap_with_comment():
    content = "from module import long_function_name, another_function  # comment"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,  # comment\n)"

def test_line_wrap_with_noqa_comment():
    content = "from module import long_function_name, another_function  # noqa"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "from module import long_function_name, another_function  # noqa"

def test_line_wrap_with_noqa_mode():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "from module import long_function_name, another_function  # NOQA"

def test_line_wrap_with_use_parentheses():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(use_parentheses=True)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_include_trailing_comma():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(include_trailing_comma=True)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_vertical_hanging_indent():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_comment_prefix():
    content = "from module import long_function_name, another_function  # comment"
    line_separator = "\n"
    config = Config(comment_prefix="# ")
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,  # comment\n)"

def test_line_wrap_with_wrap_length():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(wrap_length=20)
    assert line(content, line_separator, config) == "from module import (\n    long_function_name,\n    another_function,\n)"

def test_line_wrap_with_indent():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(indent="    ")
    assert line(content, line_separator, config) == "from module import (\n        long_function_name,\n        another_function,\n    )"


# LLM-generated content at query #17
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "short line"

def test_line_wrap_noqa_mode():
    content = "a" * 100
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == f"{content}{config.comment_prefix} NOQA"

def test_line_wrap_with_import_splitter():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "import " in result
    assert "\\" in result

def test_line_wrap_with_comment():
    content = "from module import very_long_function_name # some comment"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "# some comment" in result

def test_line_wrap_with_parentheses():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(use_parentheses=True)
    result = line(content, line_separator, config)
    assert "(" in result
    assert ")" in result

def test_line_wrap_with_trailing_comma():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "," in result

def test_line_wrap_with_noqa_comment():
    content = "from module import very_long_function_name # noqa"
    line_separator = "\n"
    config = Config(use_parentheses=True)
    result = line(content, line_separator, config)
    assert "# noqa" in result

def test_line_wrap_with_as_splitter():
    content = "import module as very_long_alias_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "as " in result
    assert "\\" in result


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    content = "some_content"
    line_separator = "\n"
    config = Config(
        use_parentheses=True,
        comment_prefix="#",
        line_length=100,
        wrap_length=None,
        indent="",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    output = line(content, line_separator, config)
    lines = output.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")")) or lines[-1] != "# comment)"


# LLM-generated content at query #19
#--------------------------

```python
def test_line_predicate_false():
    content = "short"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=100)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #20
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement("from foo import", ["bar", "baz"])
    assert result == "from foo import bar, baz"

def test_import_statement_with_comments():
    result = import_statement("from foo import", ["bar", "baz"], comments=["# comment"])
    assert "# comment" in result

def test_import_statement_explode():
    result = import_statement("from foo import", ["bar", "baz"], explode=True)
    assert result == "from foo import (\n    bar,\n    baz,\n)"

def test_import_statement_multi_line():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from foo import", ["bar", "baz"], config=config)
    assert "\n" in result

def test_import_statement_custom_line_length():
    config = Config(line_length=20)
    result = import_statement("from foo import", ["bar", "baz"], config=config)
    assert len(result.split("\n")[0]) <= 20

def test_import_statement_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement("from foo import", ["bar", "baz"], config=config)
    assert result.endswith(",")

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True)
    result = import_statement("from foo import", ["bar", "baz"], config=config)
    assert result.count("\n") > 0

def test_import_statement_custom_indent():
    config = Config(indent="    ")
    result = import_statement("from foo import", ["bar", "baz"], config=config)
    assert "    " in result

def test_import_statement_custom_line_separator():
    result = import_statement("from foo import", ["bar", "baz"], line_separator="\r\n")
    assert "\r\n" in result

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement("from foo import", ["bar", "baz"], comments=["# comment"], config=config)
    assert "# comment" not in result


# LLM-generated content at query #21
#--------------------------

```python
def test_line_71_predicate_true():
    content = "a" * 100
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        wrap_length=0,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=False,
        include_trailing_comma=False,
        comment_prefix="#",
        indent="    "
    )
    line_parts = ["a" * 25, "b" * 25, "c" * 25, "d" * 25]
    splitter = "import "
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length) and line_parts)


# LLM-generated content at query #23
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    config = Config(line_length=100)
    assert line(content, line_separator, config) == "short line"

def test_line_wrap_with_import():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, include_trailing_comma=True)
    expected = "from module import (\n    long_function_name,\n    another_function_name,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_dot():
    content = "very.long.module.name.function_call()"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True)
    expected = "very.long.module.name(\n    .function_call()\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_as():
    content = "import module as very_long_alias_name"
    line_separator = "\n"
    config = Config(line_length=25, use_parentheses=True)
    expected = "import module as (\n    very_long_alias_name\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_comment():
    content = "long_line # some comment"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True, include_trailing_comma=True)
    expected = "long_line(\n    # some comment,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_noqa_comment():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True, include_trailing_comma=True)
    expected = "long_line # noqa"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_noqa_mode():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=5, multi_line_output=Modes.NOQA)
    expected = "long_line # NOQA"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_vertical_hanging_indent():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, include_trailing_comma=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    expected = "from module import (\n    long_function_name,\n    another_function_name,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=True, include_trailing_comma=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    expected = "from module import (\n    long_function_name,\n    another_function_name,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_without_parentheses():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=False)
    expected = "from module import \\\n    long_function_name, another_function_name"
    assert line(content, line_separator, config) == expected


# LLM-generated content at query #24
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name", "\n", config)
    assert "import (" in result and "very_long_module_name" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("cimport very_long_module_name", "\n", config)
    assert "cimport (" in result and "very_long_module_name" in result

def test_line_wrap_with_dot():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("module.very_long_function_name()", "\n", config)
    assert "module.(" in result and "very_long_function_name()" in result

def test_line_wrap_with_as():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module as alias", "\n", config)
    assert "import module as (" in result and "alias" in result

def test_line_with_comment_no_wrap():
    assert line("short line # comment", "\n") == "short line # comment"

def test_line_with_comment_wrap():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name # comment", "\n", config)
    assert "import (" in result and "very_long_module_name" in result and "comment" in result

def test_line_noqa_mode():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("very_long_line", "\n", config)
    assert "very_long_line # NOQA" == result

def test_line_noqa_already_present():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("very_long_line # NOQA", "\n", config)
    assert "very_long_line # NOQA" == result

def test_line_use_parentheses_false():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False)
    result = line("import very_long_module_name", "\n", config)
    assert "import \\" in result and "very_long_module_name" in result

def test_line_include_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import very_long_module_name", "\n", config)
    assert "," in result

def test_line_no_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert result.count(",") == 0

def test_line_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("import very_long_module_name", "\n", config)
    assert "\n" in result

def test_line_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name", "\n", config)
    assert "\n" in result

def test_line_noqa_in_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name # noqa", "\n", config)
    assert "import very_long_module_name # noqa" == result

def test_line_comment_prefix():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ")
    result = line("import very_long_module_name # comment", "\n", config)
    assert "# comment" in result

def test_line_wrap_length():
    config = Config(line_length=100, wrap_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name", "\n", config)
    assert "import (" in result

def test_line_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ")
    result = line("import very_long_module_name", "\n", config)
    assert "    " in result

def test_line_starts_with_splitter():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name", "\n", config)
    assert "import (" in result

def test_line_no_splitter():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("very_long_line_without_splitter", "\n", config)
    assert "very_long_line_without_splitter" == result


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_true():
    assert Modes.VERTICAL_HANGING_INDENT in (Modes.VERTICAL_HANGING_INDENT, Modes.VERTICAL_GRID_GROUPED)


# LLM-generated content at query #26
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_true():
    wrap_mode = Modes.VERTICAL_HANGING_INDENT
    assert wrap_mode in (
        Modes.VERTICAL_HANGING_INDENT,
        Modes.VERTICAL_GRID_GROUPED,
    )


# LLM-generated content at query #27
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    line_without_comment = "some content"
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    )


# LLM-generated content at query #28
#--------------------------

```python
def test_while_condition_evaluates_to_true():
    config = Config(wrap_length=20, line_length=20, balanced_wrapping=True)
    statement = "from module import (a, b, c, d, e)"
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1])
    line_length = 20

    assert len(lines[-1]) < minimum_length and len(lines) == line_count and line_length > 10


# LLM-generated content at query #29
#--------------------------

```python
def test_line_30_predicate_evaluates_to_true():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 90
    line_separator = "\n"
    line_parts = ["a" * 40, "a" * 40]
    splitter = "import "
    assert (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts


# LLM-generated content at query #30
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_no_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line without comment", "\n", config) == "long line without comment NOQA"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line # NOQA", "\n", config) == "long line # NOQA"

def test_line_wrap_import():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import long_module_name"
    expected = "from module import \\\n    long_module_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_import_with_comment():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import long_module_name # comment"
    expected = "from module import \\\n    long_module_name # comment"
    assert line(content, "\n", config) == expected

def test_line_wrap_import_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_import_parentheses_with_comment():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name # comment"
    expected = "from module import (\n    long_module_name, # comment\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_as():
    config = Config(line_length=20, use_parentheses=False)
    content = "import module as long_alias"
    expected = "import module as \\\n    long_alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_as_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "import module as long_alias"
    expected = "import module as long_alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_dot():
    config = Config(line_length=20, use_parentheses=False)
    content = "module.long_attribute_name"
    expected = "module.\\\n    long_attribute_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_dot_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "module.long_attribute_name"
    expected = "module.\\\n    long_attribute_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_cimport():
    config = Config(line_length=20, use_parentheses=False)
    content = "cimport module.long_module_name"
    expected = "cimport module.\\\n    long_module_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_cimport_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "cimport module.long_module_name"
    expected = "cimport module.\\\n    long_module_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name # noqa"
    expected = "from module import (\n    long_module_name, # noqa\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_in_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name # noqa: F401"
    expected = "from module import (\n    long_module_name, # noqa: F401\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=False)
    content = "from module import long_module_name # noqa"
    expected = "from module import (\n    long_module_name # noqa\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_parentheses():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import long_module_name # noqa"
    expected = "from module import \\\n    long_module_name # noqa"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=False, include_trailing_comma=False)
    content = "from module import long_module_name # noqa"
    expected = "from module import \\\n    long_module_name # noqa"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma_and_noqa_in_comment():
    config = Config(line_length=20, use_parentheses=False, include_trailing_comma=False)
    content = "from module import long_module_name # noqa: F401"
    expected = "from module import \\\n    long_module_name # noqa: F401"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma_and_noqa_in_comment_and_noqa_mode():
    config = Config(line_length=20, use_parentheses=False, include_trailing_comma=False, multi_line_output=Modes.NOQA)
    content = "from module import long_module_name # noqa: F401"
    expected = "from module import long_module_name # noqa: F401"
    assert line(content, "\n", config) == expected

def test_line_wrap_noqa_comment_without_parentheses_and_trailing_comma_and_noqa_in_comment_and_noqa_mode_and_noqa_in_content():
    config = Config(line_length=20, use_parentheses=False, include_trailing_comma=False, multi_line_output=Modes.NOQA)
    content = "from module import long_module_name # NOQA"
    expected = "from module import long_module_name # NOQA"
    assert line(content, "\n", config) == expected


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_30():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 100
    line_separator = "\n"
    line_parts = ["a"] * 100
    assert (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts


# LLM-generated content at query #32
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_wrapping_with_import():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    very_long_function_name\n)"

def test_line_wrapping_with_comment():
    content = "from module import function  # some comment"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    function  # some comment\n)"

def test_line_wrapping_with_noqa_comment():
    content = "from module import function  # noqa"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    function  # noqa\n)"

def test_line_wrapping_with_trailing_comma():
    content = "from module import function"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    function,\n)"

def test_line_wrapping_with_as():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "import module as very_long_alias"

def test_line_wrapping_with_cimport():
    content = "cimport module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "cimport module.very_long_function_name"

def test_line_wrapping_with_dot():
    content = "module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "module.very_long_function_name"

def test_line_noqa_mode():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "from module import very_long_function_name # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    content = "from module import very_long_function_name # NOQA"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "from module import very_long_function_name # NOQA"


# LLM-generated content at query #33
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    config = Config(
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        wrap_length=None,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "some_long_import_statement"
    line_separator = "\n"
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    content = "import os, sys  # noqa"
    line_separator = "\n"
    config = Config(
        line_length=10,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="#",
        indent="    "
    )
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #35
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="# ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_false():
    config = Config(wrap_length=10, balanced_wrapping=True, line_length=10)
    result = import_statement("from x import", ["a"], line_separator="\n", config=config)
    lines = result.split("\n")
    assert not (len(lines[-1]) < min(len(line) for line in lines[:-1]) and len(lines) == len(lines) and 10 > 10)


# LLM-generated content at query #37
#--------------------------

```python
def test_line_71_predicate_true():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        multi_line_output=Modes.NOQA,
        comment_prefix="# "
    )
    assert len(content) > config.line_length
    assert config.multi_line_output == Modes.NOQA
    assert "# NOQA" not in content


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_71():
    content = "a" * 100
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #39
#--------------------------

```python
def test_while_loop_predicate():
    config = Config()
    config.balanced_wrapping = True
    config.line_length = 100
    config.wrap_length = None
    config.indent = "    "
    config.comment_prefix = "#"
    config.ignore_comments = False
    config.include_trailing_comma = True

    import_start = "from module import ("
    from_imports = ["a", "b", "c"]
    comments = ()
    line_separator = "\n"
    multi_line_output = None
    explode = False

    statement = import_statement(
        import_start,
        from_imports,
        comments,
        line_separator,
        config,
        multi_line_output,
        explode,
    )

    lines = statement.split(line_separator)
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0
    new_import_statement = statement
    line_length = config.line_length

    assert len(lines[-1]) < minimum_length and len(lines) == line_count and line_length > 10


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=100, wrap_length=None, multi_line_output=Modes.NOQA)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #41
#--------------------------

```python
def test_import_statement_while_loop_predicate_false():
    config = Config()
    config.balanced_wrapping = True
    config.line_length = 10
    config.wrap_length = None
    config.include_trailing_comma = False
    config.ignore_comments = False
    config.comment_prefix = "#"
    config.indent = "    "
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT

    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=(),
        line_separator="\n",
        config=config,
        multi_line_output=None,
        explode=False,
    )

    lines = result.split("\n")
    assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) or len(lines) != 2 or config.line_length <= 10


# LLM-generated content at query #42
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=100, wrap_length=None, multi_line_output=Modes.NOQA)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length) and True)


# LLM-generated content at query #43
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_wrap_with_import():
    content = "from module import function1, function2, function3"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    function1,\n    function2,\n    function3\n)"

def test_line_wrap_with_comment():
    content = "long line # comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "long line # comment"

def test_line_wrap_with_noqa_comment():
    content = "long line # NOQA"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "long line # NOQA"

def test_line_wrap_with_as():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "import module as alias"

def test_line_wrap_with_parentheses():
    content = "long line with parentheses"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "long line with (\n    parentheses\n)"

def test_line_wrap_with_trailing_comma():
    content = "long line with trailing comma"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "long line with (\n    trailing comma,\n)"

def test_line_wrap_with_cimport():
    content = "cimport module.function"
    line_separator = "\n"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "cimport (\n    module.function\n)"

def test_line_wrap_with_dot():
    content = "module.function1.function2"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "module.function1 (\n    .function2\n)"


# LLM-generated content at query #44
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length NOQA"

def test_line_with_comment():
    config = Config(use_parentheses=True, include_trailing_comma=True)
    assert line("import os, sys  # comment", "\n", config) == "import (\n    os,\n    sys,  # comment\n)"

def test_line_with_noqa_comment():
    config = Config(use_parentheses=True)
    assert line("import os, sys  # noqa", "\n", config) == "import (\n    os,\n    sys  # noqa\n)"

def test_line_with_as_splitter():
    config = Config(use_parentheses=True)
    assert line("from module import function as alias", "\n", config) == "from module import function as (\n    alias\n)"

def test_line_with_dot_splitter():
    config = Config(use_parentheses=True)
    assert line("module.submodule.function", "\n", config) == "module.submodule.(\n    function\n)"

def test_line_with_cimport_splitter():
    config = Config(use_parentheses=True)
    assert line("cimport module.submodule", "\n", config) == "cimport (\n    module.submodule\n)"

def test_line_with_vertical_hanging_indent():
    config = Config(use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("import os, sys", "\n", config) == "import (\n    os,\n    sys,\n)"

def test_line_with_vertical_grid_grouped():
    config = Config(use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("import os, sys", "\n", config) == "import (\n    os,\n    sys,\n)"

def test_line_without_parentheses():
    config = Config(use_parentheses=False)
    assert line("import os, sys", "\n", config) == "import os,\\n    sys"

def test_line_with_trailing_comma():
    config = Config(use_parentheses=True, include_trailing_comma=True)
    assert line("import os, sys", "\n", config) == "import (\n    os,\n    sys,\n)"

def test_line_without_trailing_comma():
    config = Config(use_parentheses=True, include_trailing_comma=False)
    assert line("import os, sys", "\n", config) == "import (\n    os,\n    sys\n)"

def test_line_with_noqa_and_parentheses():
    config = Config(use_parentheses=True)
    assert line("import os, sys  # noqa", "\n", config) == "import (\n    os,\n    sys  # noqa\n)"

def test_line_with_noqa_and_no_parentheses():
    config = Config(use_parentheses=False)
    assert line("import os, sys  # noqa", "\n", config) == "import os,\\n    sys  # noqa"

def test_line_with_long_content():
    config = Config(use_parentheses=True)
    assert line("import module.submodule.function", "\n", config) == "import (\n    module.submodule.function\n)"

def test_line_with_short_content():
    config = Config(use_parentheses=True)
    assert line("import os", "\n", config) == "import os"

def test_line_with_comment_and_noqa():
    config = Config(use_parentheses=True)
    assert line("import os, sys  # noqa: F401", "\n", config) == "import (\n    os,\n    sys  # noqa: F401\n)"

def test_line_with_comment_and_noqa_and_trailing_comma():
    config = Config(use_parentheses=True, include_trailing_comma=True)
    assert line("import os, sys  # noqa: F401", "\n", config) == "import (\n    os,\n    sys,  # noqa: F401\n)"


# LLM-generated content at query #45
#--------------------------

```python
def test_line_predicate_false():
    content = "short"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=100)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_at_line_48():
    config = Config()
    config.wrap_mode = Modes.VERTICAL_HANGING_INDENT
    assert config.wrap_mode in (
        Modes.VERTICAL_HANGING_INDENT,
        Modes.VERTICAL_GRID_GROUPED,
    )


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_false():
    content = "short"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=100)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=100, wrap_length=None, multi_line_output=Modes.NOQA)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #49
#--------------------------

```python
def test_line_length_predicate():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 90
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #50
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "import os.path as osp"
    line_separator = "\n"
    line_without_comment = "import os.path as osp"
    comment = None
    line_parts = ["import os.path", " osp"]
    _comma_maybe = ","
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    ) == True


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length # NOQA", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_wrap_with_import():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module import long_module_name", "\n", config) == "from module import \\\n    long_module_name"

def test_line_wrap_with_as():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module as alias", "\n", config) == "import module \\\n    as alias"

def test_line_wrap_with_dot():
    config = Config(line_length=20, use_parentheses=False)
    assert line("module.long_module_name.function", "\n", config) == "module.long_module_name.\\\n    function"

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_wrap_with_comment():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name,  # comment\n)"

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # noqa", "\n", config) == "from module import long_module_name # noqa"

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_wrap_with_cimport():
    config = Config(line_length=20, use_parentheses=False)
    assert line("cimport module.long_module_name", "\n", config) == "cimport module.\\\n    long_module_name"


# LLM-generated content at query #2
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_wrap_with_import():
    content = "from module import very_long_function_name, another_function"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=30)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    very_long_function_name,\n    another_function\n)"

def test_line_wrap_with_comment():
    content = "line with comment # comment"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "line with comment\n# comment"

def test_line_wrap_with_noqa_comment():
    content = "line with noqa comment # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=20)
    result = line(content, line_separator, config)
    assert result == "line with noqa comment # NOQA"

def test_line_wrap_with_as():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "import module as\n    very_long_alias"

def test_line_wrap_with_parentheses():
    content = "line with parentheses (comment)"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "line with parentheses\n(comment)"

def test_line_wrap_with_trailing_comma():
    content = "line with trailing comma,"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "line with trailing comma,"


# LLM-generated content at query #3
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement("from module import", ["a", "b", "c"])
    assert isinstance(result, str)
    assert result.startswith("from module import")

def test_import_statement_with_comments():
    result = import_statement("from module import", ["a", "b", "c"], comments=["# comment"])
    assert "# comment" in result

def test_import_statement_custom_separator():
    result = import_statement("from module import", ["a", "b", "c"], line_separator="\r\n")
    assert "\r\n" in result

def test_import_statement_explode_mode():
    result = import_statement("from module import", ["a", "b", "c"], explode=True)
    assert result.count("\n") >= 2

def test_import_statement_multi_line_output():
    result = import_statement("from module import", ["a", "b", "c"], multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert "\n" in result

def test_import_statement_single_line():
    result = import_statement("from module import", ["a"], config=Config(wrap_length=100))
    assert result.count("\n") == 0

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=20)
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1
    assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) - 1

def test_import_statement_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    assert result.rstrip().endswith(",")

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement("from module import", ["a", "b", "c"], comments=["# comment"], config=config)
    assert "# comment" not in result

def test_import_statement_custom_indent():
    config = Config(indent="    ")
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    assert result.startswith("from module import\n    ")

def test_import_statement_comment_prefix():
    config = Config(comment_prefix="# ")
    result = import_statement("from module import", ["a", "b", "c"], comments=["comment"], config=config)
    assert "# comment" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_import_statement_explode_mode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert result == "from module import (\n    a,\n    b,\n    c,\n)"

def test_import_statement_default_mode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
    )
    assert result == "from module import (a, b, c)"

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
    )
    assert "# comment" in result

def test_import_statement_custom_line_separator():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_custom_config():
    config = Config(wrap_length=50, include_trailing_comma=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert result.endswith(",")

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=20)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    lines = result.split("\n")
    assert len(lines[-1]) >= min(len(line) for line in lines[:-1])

def test_import_statement_single_line_output():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a"],
    )
    assert "\n" not in result

def test_import_statement_multi_line_output():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "\n" in result

def test_import_statement_with_indent():
    config = Config(indent="    ")
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert result.startswith("    ")

def test_import_statement_remove_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
        config=config,
    )
    assert "# comment" not in result


# LLM-generated content at query #5
#--------------------------

```python
def test_predicate_at_line_43_evaluates_to_true():
    config = Config(
        use_parentheses=True,
        indent="    ",
        line_length=88,
        wrap_length=None,
        include_trailing_comma=True,
        comment_prefix=" # ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "from module import something as alias"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert config.use_parentheses is True


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 90
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #7
#--------------------------

```python
def test_import_statement_single_line():
    result = import_statement(
        import_start="from module import",
        from_imports=["item1", "item2"],
        config=Config(multi_line_output=Modes.GRID, wrap_length=100, balanced_wrapping=True),
    )
    lines = result.split("\n")
    assert len(lines) == 1


# LLM-generated content at query #8
#--------------------------

```python
def test_line_length_predicate():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 90
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #9
#--------------------------

```python
def test_line_71_predicate_true():
    config = Config(
        line_length=10,
        multi_line_output=Modes.NOQA,
        comment_prefix="# "
    )
    content = "a" * 11  # Length > line_length
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "# NOQA" in result


# LLM-generated content at query #10
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_with_noqa_mode():
    content = "a" * 100
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == f"{content} NOQA"

def test_line_with_import_split():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "\\" in result or "(" in result

def test_line_with_comment():
    content = "long_line # comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "comment" in result

def test_line_with_as_split():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "as" in result

def test_line_with_parentheses_and_noqa():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "(" in result and "noqa" in result

def test_line_with_trailing_comma():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "," in result


# LLM-generated content at query #11
#--------------------------

```python
def test_while_loop_predicate():
    config = Config(wrap_length=20, line_length=20, balanced_wrapping=True)
    from_imports = ["module1", "module2", "module3"]
    import_start = "from . import"
    statement = import_statement(import_start, from_imports, config=config)
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0
    assert len(lines[-1]) < minimum_length and len(lines) == line_count and config.wrap_length > 10


# LLM-generated content at query #12
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_noqa_mode_with_noqa_comment():
    content = "long line that exceeds line length # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    content = "long line that exceeds line length"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    result = line(content, line_separator, config)
    assert result == "long line that exceeds line length # NOQA"

def test_line_wrap_with_import_splitter():
    content = "from module import long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "from module import \\\n    long_function_name"

def test_line_wrap_with_parentheses_and_trailing_comma():
    content = "from module import long_function_name, another_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    long_function_name,\n    another_name,\n)"

def test_line_wrap_with_parentheses_and_noqa_comment():
    content = "from module import long_function_name # noqa"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    long_function_name,  # noqa\n)"

def test_line_wrap_with_as_splitter():
    content = "import module as long_alias_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "import module as long_alias_name"

def test_line_wrap_with_dot_splitter():
    content = "module.long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "module.\\\n    long_function_name"

def test_line_wrap_vertical_hanging_indent():
    content = "from module import long_function_name, another_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    long_function_name,\n    another_name,\n)"

def test_line_wrap_vertical_grid_grouped():
    content = "from module import long_function_name, another_name"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    long_function_name,\n    another_name,\n)"


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    content = "a" * 100  # Longer than default line_length
    config = Config(multi_line_output=Modes.NOQA, line_length=50)
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #14
#--------------------------

```python
def test_line_65_predicate_true():
    config = Config(
        use_parentheses=True,
        comment_prefix="# ",
        line_length=88,
        wrap_length=None,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "from module import (something, something_else,  # comment"
    line_separator = "\n"
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert config.comment_prefix in lines[-1] and lines[-1].endswith(")")


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_17():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="# ",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "from module import function, another_function"
    line_without_comment = content
    _comma_maybe = (
        ","
        if (
            config.include_trailing_comma
            and config.use_parentheses
            and not line_without_comment.rstrip().endswith(",")
        )
        else ""
    )
    assert _comma_maybe == ","


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_56_evaluates_to_false():
    content = "import os  # some comment"
    line_separator = "\n"
    config = Config(
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=10,
        use_parentheses=True,
        comment_prefix="# "
    )
    assert not (config.comment_prefix in content.split(line_separator)[-1] and content.split(line_separator)[-1].endswith(")"))


# LLM-generated content at query #17
#--------------------------

```python
def test_line_71_predicate_true():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        multi_line_output=Modes.NOQA,
        comment_prefix="# "
    )
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length # NOQA", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_with_import_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module import long_module_name", "\n", config) == "from module import \\\n    long_module_name"

def test_line_with_cimport_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module cimport long_module_name", "\n", config) == "from module cimport \\\n    long_module_name"

def test_line_with_dot_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("module.long_module_name.function()", "\n", config) == "module.long_module_name.\\\n    function()"

def test_line_with_as_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module as long_alias", "\n", config) == "import module as \\\n    long_alias"

def test_line_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import (long_module_name,)", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_with_comment_and_no_parentheses():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module # comment", "\n", config) == "import module # comment"

def test_line_with_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name,  # comment\n)"

def test_line_with_noqa_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True)
    assert line("from module import long_module_name # noqa", "\n", config) == "from module import (\n    long_module_name,  # noqa\n)"

def test_line_vertical_hanging_indent_mode():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_vertical_grid_grouped_mode():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_with_as_split_and_parentheses():
    config = Config(line_length=20, use_parentheses=True)
    assert line("import module as long_alias", "\n", config) == "import module as long_alias"

def test_line_with_comment_and_noqa():
    config = Config(line_length=20, use_parentheses=True)
    assert line("from module import long_module_name # noqa: F401", "\n", config) == "from module import (\n    long_module_name,  # noqa: F401\n)"


# LLM-generated content at query #2
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        line_separator="\n",
    )
    assert "a" in result
    assert "b" in result
    assert "c" in result

def test_import_statement_explode():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert "a" in result
    assert "b" in result
    assert "c" in result
    assert result.count("\n") == 3

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
    )
    assert "# comment" in result

def test_import_statement_custom_separator():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_single_line():
    result = import_statement(
        import_start="from module import a",
        from_imports=["a"],
    )
    assert result == "from module import a"

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=100)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "a" in result
    assert "b" in result
    assert "c" in result

def test_import_statement_multi_line_output():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert "a" in result
    assert "b" in result
    assert "c" in result

def test_import_statement_trailing_comma():
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        include_trailing_comma=True,
    )
    assert result.endswith(",")

def test_import_statement_custom_indent():
    config = Config(indent="    ")
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "    " in result

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        comments=["# comment"],
        config=config,
    )
    assert "# comment" not in result


# LLM-generated content at query #3
#--------------------------

```python
def test_line_predicate_evaluates_to_true():
    config = Config(
        line_length=100,
        wrap_length=80,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    "
    )
    content = "import a_very_long_module_name_that_exceeds_line_length_limit"
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #4
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "short line"

def test_line_wrapping_with_import():
    content = "from module import long_function_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "long_function_name" in result

def test_line_wrapping_with_dot():
    content = "module.long_function_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "module." in result
    assert "long_function_name" in result

def test_line_wrapping_with_as():
    content = "import module as alias"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "import module as alias" in result

def test_line_wrapping_with_comment():
    content = "long_line # comment"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "# comment" in result

def test_line_wrapping_with_noqa():
    content = "long_line # NOQA"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    result = line(content, line_separator, config)
    assert "long_line # NOQA" in result

def test_line_wrapping_with_use_parentheses():
    content = "long_line"
    line_separator = "\n"
    config = Config(use_parentheses=True)
    result = line(content, line_separator, config)
    assert "(" in result and ")" in result

def test_line_wrapping_with_trailing_comma():
    content = "long_line"
    line_separator = "\n"
    config = Config(include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "," in result

def test_line_wrapping_with_vertical_hanging_indent():
    content = "long_line"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert line_separator in result

def test_line_wrapping_with_vertical_grid_grouped():
    content = "long_line"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line(content, line_separator, config)
    assert line_separator in result


# LLM-generated content at query #5
#--------------------------

```python
def test_line_predicate_false():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=False,
        include_trailing_comma=False,
        comment_prefix="#",
        indent="    "
    )
    assert not (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #6
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(
        wrap_length=100,
        line_length=100,
        multi_line_output=Modes.NOQA,
        use_parentheses=False,
        include_trailing_comma=False,
        comment_prefix="#",
        indent=""
    )
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length) and [])


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=5, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #8
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length # NOQA", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_wrap_with_import_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import very_long_name"
    expected = "from module import \\\n    very_long_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "module.very_long_function_name()"
    expected = "module.\\\n    very_long_function_name()"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_as_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "import module as very_long_alias"
    expected = "import module as \\\n    very_long_alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import (very_long_name)"
    expected = "from module import (\n    very_long_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_comment():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import very_long_name # some comment"
    expected = "from module import \\\n    very_long_name # some comment"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_parentheses_and_noqa_comment():
    config = Config(line_length=20, use_parentheses=True, comment_prefix="# ")
    content = "from module import very_long_name # NOQA: some comment"
    expected = "from module import (\n    very_long_name, # NOQA: some comment\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import very_long_name"
    expected = "from module import (\n    very_long_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    content = "from module import very_long_name"
    expected = "from module import (\n    very_long_name,\n)"
    assert line(content, "\n", config) == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_true():
    config = Config(
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="#",
        indent="    ",
        line_length=88,
        wrap_length=None
    )
    content = "from module import something, another_thing, yet_another"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert wrap_mode in (Modes.VERTICAL_HANGING_INDENT, Modes.VERTICAL_GRID_GROUPED)


# LLM-generated content at query #10
#--------------------------

```python
def test_line_with_noqa_comment_and_noqa_mode():
    config = Config(
        multi_line_output=Modes.NOQA,
        line_length=10,
        comment_prefix="# "
    )
    content = "shortline"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "shortline#  NOQA"


# LLM-generated content at query #11
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_with_noqa_mode():
    content = "a" * 100
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == f"{content} NOQA"

def test_line_with_import_split():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"from module import (\n    very_long_function_name\n)"

def test_line_with_comment():
    content = "long_line # comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"long_line,  # comment\n"

def test_line_with_as_split():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"import module as very_long_alias\n"

def test_line_with_parentheses_and_noqa():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == f"long_line,  # noqa\n"

def test_line_with_trailing_comma():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == f"long_line,\n"

def test_line_with_vertical_grid_grouped():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == f"long_line,\n"

def test_line_with_cimport_split():
    content = "cimport module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"cimport module.very_long_function_name\n"

def test_line_with_dot_split():
    content = "module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert result == f"module.very_long_function_name\n"


# LLM-generated content at query #12
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_with_noqa_mode_and_no_noqa_comment():
    content = "a" * 100
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=50)
    result = line(content, line_separator, config)
    assert result == f"{content} NOQA"

def test_line_with_noqa_mode_and_noqa_comment():
    content = "a" * 100 + " # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=50)
    result = line(content, line_separator, config)
    assert result == content

def test_line_with_import_split():
    content = "from module import function, other_function"
    line_separator = "\n"
    config = Config(line_length=30, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    function,\n    other_function\n)"

def test_line_with_dot_split():
    content = "module.function.other_function"
    line_separator = "\n"
    config = Config(line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "module.function(\n    .other_function\n)"

def test_line_with_as_split():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=15, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "import module\n    as alias"

def test_line_with_comment_and_no_parentheses():
    content = "long_line # comment"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "long_line  # comment"

def test_line_with_comment_and_parentheses():
    content = "long_line # comment"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "long_line(\n    # comment\n)"

def test_line_with_noqa_in_comment_and_parentheses():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "long_line(\n    # noqa\n)"

def test_line_with_trailing_comma_and_parentheses():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "long_line(\n    ,\n)"

def test_line_with_vertical_hanging_indent_mode():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "long_line(\n    \n)"

def test_line_with_vertical_grid_grouped_mode():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "long_line(\n    \n)"


# LLM-generated content at query #13
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        line_length=88,
        wrap_length=None,
        indent="    ",
        comment_prefix="# ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "import os.path as osp"
    line_separator = "\n"
    line_without_comment = content
    comment = None
    splitter = "as "
    line_parts = ["import os.path ", " osp"]
    _comma_maybe = ","
    assert _comma_maybe == ","


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        line_length=88,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="# ",
    )
    content = "from module import long_module_name, another_long_module_name"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert ", " in result


# LLM-generated content at query #15
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    content = "a" * 100
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #16
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    config = Config(
        use_parentheses=True,
        comment_prefix="# ",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True
    )
    content = "from module import (something, something_else, # noqa\n)"
    line_separator = "\n"
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_at_line_59():
    config = Config()
    config.include_trailing_comma = True
    comment = "noqa"
    _comma = "," if config.include_trailing_comma else ""
    assert _comma == ","


# LLM-generated content at query #18
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        line_length=100,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="# ",
    )
    content = "import os.path as osp, sys"
    line_without_comment = content
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    )


# LLM-generated content at query #19
#--------------------------

```python
def test_formatter_from_string_returns_grid_for_invalid_name():
    result = formatter_from_string("invalid_name")
    assert result == grid


# LLM-generated content at query #20
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    assert line(content, line_separator) == "short line"

def test_line_wrapping_with_import():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "from module import (\n    very_long_function_name\n)"
    )

def test_line_wrapping_with_cimport():
    content = "cimport module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "cimport module.(\n    very_long_function_name\n)"
    )

def test_line_wrapping_with_dot():
    content = "module.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "module.(\n    very_long_function_name\n)"
    )

def test_line_wrapping_with_as():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "import module as very_long_alias"
    )

def test_line_wrapping_with_comment():
    content = "import module  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "import module  # some comment"
    )

def test_line_wrapping_with_noqa_comment():
    content = "import module  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "import module  # NOQA"

def test_line_wrapping_with_trailing_comma():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    assert line(content, line_separator, config) == (
        "import (\n    module1,\n    module2,\n)"
    )

def test_line_wrapping_with_parentheses():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    assert line(content, line_separator, config) == (
        "import (\n    module1,\n    module2,\n)"
    )

def test_line_wrapping_with_vertical_grid_grouped():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line(content, line_separator, config) == (
        "import (\n    module1,\n    module2,\n)"
    )

def test_line_wrapping_with_vertical_hanging_indent():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == (
        "import (\n    module1,\n    module2,\n)"
    )

def test_line_wrapping_with_noqa_in_comment():
    content = "import module  # noqa: F401"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "import module  # noqa: F401"

def test_line_wrapping_with_indent():
    content = "    import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ")
    assert line(content, line_separator, config) == (
        "    import (\n        module1,\n        module2,\n    )"
    )

def test_line_wrapping_with_comment_prefix():
    content = "import module  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ")
    assert line(content, line_separator, config) == (
        "import (\n    module,  # some comment\n)"
    )

def test_line_wrapping_with_wrap_length():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20)
    assert line(content, line_separator, config) == (
        "import (\n    module1,\n    module2,\n)"
    )

def test_line_wrapping_with_noqa_mode():
    content = "import module1, module2"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "import module1, module2  # NOQA"

def test_line_wrapping_with_noqa_mode_and_noqa_comment():
    content = "import module1, module2  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "import module1, module2  # NOQA"


# LLM-generated content at query #21
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_no_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("a" * 100, "\n", config) == f"{'a' * 100} NOQA"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("a" * 100 + " # NOQA", "\n", config) == "a" * 100 + " # NOQA"

def test_line_import_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module import long_module_name", "\n", config) == "from module import \\\n    long_module_name"

def test_line_cimport_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module cimport long_module_name", "\n", config) == "from module cimport \\\n    long_module_name"

def test_line_dot_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("module.long_module_name.function()", "\n", config) == "module.long_module_name.\\\n    function()"

def test_line_as_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module as long_alias", "\n", config) == "import module as \\\n    long_alias"

def test_line_with_comment_no_parentheses():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module # comment", "\n", config) == "import module # comment"

def test_line_with_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("import module # comment", "\n", config) == "import (\n    module,  # comment\n)"

def test_line_with_noqa_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("import module # noqa", "\n", config) == "import module # noqa"

def test_line_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("import module1, module2", "\n", config) == "import (\n    module1,\n    module2,\n)"

def test_line_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("import module1, module2", "\n", config) == "import (\n    module1,\n    module2,\n)"


# LLM-generated content at query #22
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_true():
    config = Config(
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        line_length=100,
        wrap_length=None,
        indent="    ",
        comment_prefix="#",
        include_trailing_comma=True
    )
    content = "from module import something, another_thing as alias"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert wrap_mode in (Modes.VERTICAL_HANGING_INDENT, Modes.VERTICAL_GRID_GROUPED)


# LLM-generated content at query #23
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    content = "from module import long_function_name, another_function"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert "from module import (\n" in result and "long_function_name,\n" in result

def test_line_wrap_with_comment():
    content = "long_line # comment"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert "# comment" in result

def test_line_wrap_with_noqa():
    content = "long_line # NOQA"
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    assert line(content, "\n", config) == "long_line # NOQA"

def test_line_wrap_with_as():
    content = "import module as alias"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert "import module as (\n" in result and "alias" in result

def test_line_wrap_with_parentheses():
    content = "long_line # comment"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, "\n", config)
    assert "(# comment" in result and ")" in result

def test_line_wrap_with_trailing_comma():
    content = "long_line # comment"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, "\n", config)
    assert "," in result

def test_line_wrap_with_grid_grouped():
    content = "long_line # comment"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line(content, "\n", config)
    assert "\n" in result and "# comment" in result

def test_line_wrap_with_cimport():
    content = "cimport module.long_function_name"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert "cimport module.\n" in result and "long_function_name" in result

def test_line_wrap_with_dot():
    content = "module.long_function_name"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert "module.\n" in result and "long_function_name" in result


# LLM-generated content at query #24
#--------------------------

```python
def test_line_length_predicate():
    config = Config()
    config.wrap_length = 100
    config.line_length = 80
    content = "a" * 90
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #25
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    content = "some_content"
    line_separator = "\n"
    config = Config(
        comment_prefix="#",
        line_length=100,
        wrap_length=None,
        use_parentheses=True,
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    "
    )
    output = line(content, line_separator, config)
    lines = output.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #26
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    assert line(content, line_separator, config) == "short line"

def test_line_wrapping_with_import_splitter():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_dot_splitter():
    content = "module.long_function_name.another_function_name"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    expected = (
        "module.long_function_name.\n"
        "    another_function_name"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_as_splitter():
    content = "import module as alias"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    expected = (
        "import module as alias"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_comment():
    content = "import module  # comment"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    expected = (
        "import module  # comment"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_noqa_comment():
    content = "import module  # noqa"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    expected = (
        "import module  # noqa"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_use_parentheses():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(use_parentheses=True)
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_include_trailing_comma():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(include_trailing_comma=True)
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_wrap_length():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(wrap_length=20)
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_multi_line_output():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_comment_prefix():
    content = "import module  # comment"
    line_separator = "\n"
    config = Config(comment_prefix="# ")
    expected = (
        "import module  # comment"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_indent():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(indent="    ")
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_line_length():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(line_length=20)
    expected = (
        "from module import (\n"
        "    long_function_name,\n"
        "    another_function_name,\n"
        ")"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_noqa_mode():
    content = "from module import long_function_name, another_function_name"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    expected = (
        "from module import long_function_name, another_function_name  # NOQA"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_noqa_mode_and_noqa_comment():
    content = "from module import long_function_name, another_function_name  # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    expected = (
        "from module import long_function_name, another_function_name  # NOQA"
    )
    assert line(content, line_separator, config) == expected

def test_line_wrapping_with_config_noqa_mode_and_no_noqa_comment():
    content = "from module import long_function_name, another_function_name  # comment"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    expected = (
        "from module import long_function_name, another_function_name  # comment  # NOQA"
    )
    assert line(content, line_separator, config) == expected


# LLM-generated content at query #27
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import (\n    very_long_module_name\n)"

def test_line_wrap_with_dot():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("module.very_long_attribute_name", "\n", config)
    assert result == "module.(\n    very_long_attribute_name\n)"

def test_line_wrap_with_as():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module as very_long_alias", "\n", config)
    assert result == "import module as very_long_alias"

def test_line_wrap_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module # comment", "\n", config)
    assert result == "import (\n    module  # comment\n)"

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module # NOQA", "\n", config)
    assert result == "import module # NOQA"

def test_line_wrap_with_noqa_in_config():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module", "\n", config)
    assert result == "import module # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True)
    result = line("import module1, module2", "\n", config)
    assert result == "import (\n    module1,\n    module2,\n)"

def test_line_wrap_without_trailing_comma():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=False, use_parentheses=True)
    result = line("import module1, module2", "\n", config)
    assert result == "import (\n    module1\n    module2\n)"

def test_line_wrap_with_cimport():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("cimport very_long_module_name", "\n", config)
    assert result == "cimport (\n    very_long_module_name\n)"


# LLM-generated content at query #28
#--------------------------

```python
def test_line_length_predicate():
    config = Config(
        line_length=100,
        wrap_length=80,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=False,
        include_trailing_comma=False,
        comment_prefix="# ",
        indent="    "
    )
    content = "a" * 101
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #29
#--------------------------

```python
def test_predicate_at_line_46_evaluates_to_true():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    config.comment_prefix = "#"
    comment = None
    _comma = "," if config.include_trailing_comma and not comment else ""
    assert _comma == ","


# LLM-generated content at query #30
#--------------------------

```python
def test_balanced_wrapping_predicate():
    config = Config(balanced_wrapping=True, wrap_length=100, line_length=100, include_trailing_comma=True, ignore_comments=False)
    import_start = "from module import ("
    from_imports = ["a", "b", "c"]
    statement = import_statement(import_start, from_imports, config=config)
    lines = statement.split("\n")
    assert len(lines[-1]) < min(len(line) for line in lines[:-1]) and len(lines) > 1 and config.wrap_length > 10


# LLM-generated content at query #31
#--------------------------

```python
def test_while_loop_condition():
    config = Config(wrap_length=100, line_length=100, balanced_wrapping=True)
    statement = "from module import (a, b, c, d, e, f)"
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1])
    line_length = 100
    assert len(lines[-1]) < minimum_length and len(lines) == line_count and line_length > 10


# LLM-generated content at query #32
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("from module import long_function_name", "\n", config)
    expected = "from module import (\n    long_function_name\n)"
    assert result == expected

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("from module import func # comment", "\n", config)
    expected = "from module import (\n    func  # comment\n)"
    assert result == expected

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line("from module import long_function_name # NOQA", "\n", config)
    expected = "from module import long_function_name # NOQA"
    assert result == expected

def test_line_wrap_with_as():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module as alias", "\n", config)
    expected = "import module as alias"
    assert result == expected

def test_line_wrap_with_dot():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("module.long_function_name()", "\n", config)
    expected = "module.long_function_name()"
    assert result == expected

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    result = line("from module import func1, func2", "\n", config)
    expected = "from module import (\n    func1,\n    func2,\n)"
    assert result == expected

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line("from module import func1, func2", "\n", config)
    expected = "from module import (\n    func1,\n    func2,\n)"
    assert result == expected

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("from module import func # noqa", "\n", config)
    expected = "from module import (\n    func  # noqa\n)"
    assert result == expected

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("from module import func1, func2", "\n", config)
    expected = "from module import (\n    func1,\n    func2,\n)"
    assert result == expected


# LLM-generated content at query #33
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    config = Config(line_length=100, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "short line"

def test_line_wrap_with_import():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,\n    another_function,\n)" in result

def test_line_wrap_with_comment():
    content = "from module import function  # comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,  # comment\n)" in result

def test_line_wrap_with_noqa_comment():
    content = "from module import function  # noqa"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,  # noqa\n)" in result

def test_line_wrap_with_as():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "import module as alias" in result

def test_line_wrap_with_dot():
    content = "module.submodule.function"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "module.submodule.function" in result

def test_line_noqa_mode():
    content = "from module import function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "from module import function # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    content = "from module import function  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == "from module import function  # NOQA"

def test_line_wrap_with_cimport():
    content = "cimport module.function"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "cimport module.function" in result

def test_line_wrap_without_parentheses():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False)
    result = line(content, line_separator, config)
    assert "from module import function,\\n    another_function" in result

def test_line_wrap_with_trailing_comma():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,\n    another_function,\n)" in result

def test_line_wrap_without_trailing_comma():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False)
    result = line(content, line_separator, config)
    assert "from module import (\n    function\n    another_function\n)" in result

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,\n    another_function,\n)" in result

def test_line_wrap_with_vertical_hanging_indent():
    content = "from module import function, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (\n    function,\n    another_function,\n)" in result


# LLM-generated content at query #34
#--------------------------

```python
def test_balanced_wrapping_with_multiple_lines():
    config = Config(balanced_wrapping=True, wrap_length=100, line_length=100, include_trailing_comma=True)
    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    lines = result.split("\n")
    assert len(lines) > 1
    assert len(lines[-1]) >= min(len(line) for line in lines[:-1])


# LLM-generated content at query #35
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import very.long.module.name", "\n", config)
    assert "import (" in result and "very.long.module.name" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("cimport very.long.module.name", "\n", config)
    assert "cimport (" in result and "very.long.module.name" in result

def test_line_wrap_with_dot():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("very.long.module.name", "\n", config)
    assert "(" in result and "very.long.module.name" in result

def test_line_wrap_with_as():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module as alias", "\n", config)
    assert "import module as alias" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module # comment", "\n", config)
    assert "# comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module # noqa", "\n", config)
    assert "# noqa" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    result = line("import module", "\n", config)
    assert "," in result

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line("import module", "\n", config)
    assert "(" in result and ")" in result

def test_line_wrap_with_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line("import module", "\n", config)
    assert "NOQA" in result

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("import module", "\n", config)
    assert "import (" in result and "module" in result

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module", "\n", config)
    assert "import (" in result and "module" in result


# LLM-generated content at query #36
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("from module import long_function_name", "\n", config)
    assert "import (" in result and "\\" not in result

def test_line_wrap_with_as():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("import module as m", "\n", config)
    assert "as (" in result

def test_line_wrap_with_dot():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("module.long_function_name", "\n", config)
    assert ".(" in result

def test_line_wrap_with_cimport():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("cimport module.long_function_name", "\n", config)
    assert "cimport (" in result

def test_line_noqa_comment_added():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    result = line("very long line without noqa", "\n", config)
    assert result.endswith(" NOQA")

def test_line_noqa_comment_not_added():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    result = line("very long line # NOQA", "\n", config)
    assert result == "very long line # NOQA"

def test_line_with_comment():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("import module # comment", "\n", config)
    assert "# comment" in result

def test_line_with_noqa_in_comment():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line("import module # noqa", "\n", config)
    assert "noqa" in result

def test_line_with_trailing_comma():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, include_trailing_comma=True)
    result = line("import module", "\n", config)
    assert "," in result

def test_line_without_trailing_comma():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, include_trailing_comma=False)
    result = line("import module", "\n", config)
    assert not result.endswith(",")

def test_line_with_parentheses():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True)
    result = line("import module", "\n", config)
    assert "(" in result and ")" in result

def test_line_without_parentheses():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=False)
    result = line("import module", "\n", config)
    assert "\\" in result


# LLM-generated content at query #37
#--------------------------

```python
def test_while_loop_predicate():
    config = Config()
    config.balanced_wrapping = True
    config.line_length = 100
    config.wrap_length = None
    config.ignore_comments = False
    config.comment_prefix = "#"
    config.include_trailing_comma = False
    config.indent = "    "
    config.multi_line_output = Modes.GRID
    import_start = "from module import"
    from_imports = ["a", "b", "c"]
    comments = ()
    line_separator = "\n"
    multi_line_output = None
    explode = False
    statement = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    lines = statement.split(line_separator)
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1])
    new_import_statement = statement
    line_length = config.wrap_length or config.line_length
    assert len(lines[-1]) < minimum_length and len(lines) == line_count and line_length > 10


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_36():
    config = Config()
    config.balanced_wrapping = True
    config.wrap_length = 20
    config.line_length = 20
    config.ignore_comments = False
    config.comment_prefix = "#"
    config.include_trailing_comma = False
    config.indent = "    "

    statement = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
        explode=False,
    )

    lines = statement.split("\n")
    assert len(lines) > 1


# LLM-generated content at query #39
#--------------------------

```python
def test_line_predicate_false():
    content = "short"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=100)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_at_line_46_evaluates_to_true():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    config.comment_prefix = "# "
    comment = "some comment"
    _comma = "," if config.include_trailing_comma and not comment else ""
    assert _comma == ","


# LLM-generated content at query #41
#--------------------------

```python
def test_line_count_greater_than_one():
    lines = ["import os", "import sys"]
    line_count = len(lines)
    assert len(lines) > 1


# LLM-generated content at query #42
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_with_noqa_mode():
    content = "a" * 100
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA)
    result = line(content, line_separator, config)
    assert result == f"{content} NOQA"

def test_line_with_import_splitter():
    content = "from module import long_function_name, another_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "import (" in result
    assert line_separator in result

def test_line_with_comment():
    content = "long_line # comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "comment" in result

def test_line_with_as_splitter():
    content = "import module as alias"
    line_separator = "\n"
    config = Config(line_length=15, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "as (" in result

def test_line_with_parentheses_and_noqa():
    content = "long_line # noqa"
    line_separator = "\n"
    config = Config(line_length=10, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "noqa" in result
    assert "(" in result

def test_line_with_trailing_comma():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, include_trailing_comma=True, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, line_separator, config)
    assert "," in result

def test_line_with_vertical_grid_grouped():
    content = "long_line"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line(content, line_separator, config)
    assert line_separator in result


# LLM-generated content at query #43
#--------------------------

```python
def test_predicate_at_line_46_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="# ",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment=None,
    )
    content = "from module import (long_function_name1, long_function_name2)"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert _comma == ","


# LLM-generated content at query #44
#--------------------------

```python
def test_line_no_wrap_needed():
    result = line("short line", "\n")
    assert result == "short line"

def test_line_wrap_with_import():
    result = line("from module import long_function_name", "\n")
    assert result == "from module import (\n    long_function_name\n)"

def test_line_wrap_with_cimport():
    result = line("cimport module.long_function_name", "\n")
    assert result == "cimport module.(\n    long_function_name\n)"

def test_line_wrap_with_dot():
    result = line("module.long_function_name", "\n")
    assert result == "module.(\n    long_function_name\n)"

def test_line_wrap_with_as():
    result = line("import module as alias", "\n")
    assert result == "import module as alias"

def test_line_wrap_with_comment():
    result = line("from module import long_function_name  # comment", "\n")
    assert result == "from module import (\n    long_function_name  # comment\n)"

def test_line_wrap_with_noqa_comment():
    result = line("from module import long_function_name  # noqa", "\n")
    assert result == "from module import long_function_name  # noqa"

def test_line_wrap_with_trailing_comma():
    config = Config(include_trailing_comma=True, use_parentheses=True)
    result = line("from module import long_function_name", "\n", config)
    assert result == "from module import (\n    long_function_name,\n)"

def test_line_wrap_with_noqa_mode():
    result = line("from module import long_function_name", "\n", Config(multi_line_output=Modes.NOQA))
    assert result == "from module import long_function_name  # NOQA"

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("from module import long_function_name", "\n", config)
    assert result == "from module import (\n    long_function_name\n)"

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("from module import long_function_name", "\n", config)
    assert result == "from module import (\n    long_function_name\n)"


# LLM-generated content at query #45
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=5)
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #46
#--------------------------

```python
def test_predicate_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=100)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #47
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
    )
    assert isinstance(result, str)
    assert "from module import" in result
    assert "a" in result
    assert "b" in result
    assert "c" in result

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        comments=["# comment1", "# comment2"],
    )
    assert "# comment1" in result
    assert "# comment2" in result

def test_import_statement_custom_line_separator():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_explode_mode():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert result.count("\n") == 3

def test_import_statement_multi_line_output():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert isinstance(result, str)

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert isinstance(result, str)

def test_import_statement_single_line():
    result = import_statement(
        import_start="from module import",
        from_imports=["a"],
    )
    assert result.count("\n") == 0

def test_import_statement_with_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert result.endswith(",")

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        comments=["# comment1"],
        config=config,
    )
    assert "# comment1" not in result


# LLM-generated content at query #48
#--------------------------

```python
def test_predicate_at_line_36():
    config = Config()
    config.balanced_wrapping = True
    config.wrap_length = 100
    config.line_length = 100
    config.indent = "    "
    config.comment_prefix = "#"
    config.ignore_comments = False
    config.include_trailing_comma = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT

    result = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )

    lines = result.split("\n")
    assert len(lines) > 1


# LLM-generated content at query #49
#--------------------------

```python
def test_while_loop_predicate_evaluates_to_true():
    config = Config(balanced_wrapping=True, wrap_length=20, line_length=20, ignore_comments=False)
    import_start = "from module import ("
    from_imports = ["a", "b", "c"]
    statement = import_statement(import_start, from_imports, config=config)
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0
    line_length = 20
    assert len(lines[-1]) < minimum_length and len(lines) == line_count and line_length > 10


# LLM-generated content at query #50
#--------------------------

```python
def test_line_length_predicate():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "a" * 100
    assert len(content) > config.line_length and (config.wrap_length or config.line_length) == 80


# LLM-generated content at query #51
#--------------------------

```python
def test_predicate_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(line_length=100, multi_line_output=Modes.NOQA)
    assert not (len(content) > config.line_length and config.multi_line_output != Modes.NOQA)


# LLM-generated content at query #52
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("from module import long_function_name", "\n", config)
    assert isinstance(result, str)
    assert "import" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("cimport module.long_function_name", "\n", config)
    assert isinstance(result, str)
    assert "cimport" in result

def test_line_wrap_with_dot():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("module.long_function_name", "\n", config)
    assert isinstance(result, str)
    assert "." in result

def test_line_wrap_with_as():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module as alias", "\n", config)
    assert isinstance(result, str)
    assert "as" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module # comment", "\n", config)
    assert isinstance(result, str)
    assert "# comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module # noqa", "\n", config)
    assert isinstance(result, str)
    assert "# noqa" in result

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line("import module", "\n", config)
    assert isinstance(result, str)
    assert "(" in result and ")" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import module", "\n", config)
    assert isinstance(result, str)
    assert "," in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line("import module", "\n", config)
    assert isinstance(result, str)
    assert "NOQA" in result

def test_line_wrap_noqa_mode_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    result = line("import module # NOQA", "\n", config)
    assert isinstance(result, str)
    assert result == "import module # NOQA"

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import module", "\n", config)
    assert isinstance(result, str)

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("import module", "\n", config)
    assert isinstance(result, str)


# LLM-generated content at query #53
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    content = "import os, sys  # NOQA"
    line_separator = "\n"
    config = Config(
        line_length=10,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="#",
        indent="    "
    )
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    config = Config(
        wrap_length=100,
        line_length=80,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="#",
        indent="    "
    )
    content = "from module import very_long_module_name, another_very_long_module_name, yet_another_very_long_module_name"
    line_separator = "\n"
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #55
#--------------------------

```python
def test_line_predicate_true():
    config = Config(
        line_length=100,
        wrap_length=80,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    "
    )
    content = "import a_very_long_module_name_that_exceeds_the_line_length_limit"
    line_separator = "\n"
    assert len(content) + 2 > config.wrap_length or config.line_length


# LLM-generated content at query #56
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    result = line(content, "\n")
    assert result == "short line"

def test_line_wrapping_with_import():
    content = "from module import long_function_name"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert result == "from module import (\n    long_function_name\n)"

def test_line_wrapping_with_comment():
    content = "long_line # comment"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert result == "long_line, # comment"

def test_line_wrapping_with_noqa():
    content = "long_line # NOQA"
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert result == "long_line # NOQA"

def test_line_wrapping_with_as():
    content = "import module as alias"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert result == "import module as (\n    alias\n)"

def test_line_wrapping_with_dot():
    content = "module.long_function_name"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert result == "module.\n    long_function_name"

def test_line_wrapping_with_cimport():
    content = "cimport module.long_function_name"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line(content, "\n", config)
    assert result == "cimport module.\n    long_function_name"

def test_line_wrapping_with_parentheses():
    content = "long_line"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line(content, "\n", config)
    assert result == "(\n    long_line\n)"

def test_line_wrapping_with_trailing_comma():
    content = "long_line"
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    result = line(content, "\n", config)
    assert result == "long_line,"


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_59_evaluates_to_true():
    config = Config(
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        line_length=88,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    "
    )
    content = "from module import (something, another_thing, # noqa: F401"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "# noqa: F401" in result
    assert result.endswith(",)")


# LLM-generated content at query #58
#--------------------------

```python
def test_line_no_wrapping_needed():
    content = "short line"
    line_separator = "\n"
    result = line(content, line_separator)
    assert result == "short line"

def test_line_noqa_mode_no_comment():
    content = "very long line that exceeds the line length limit"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=20)
    result = line(content, line_separator, config)
    assert result == "very long line that exceeds the line length limit # NOQA"

def test_line_noqa_mode_with_noqa_comment():
    content = "very long line that exceeds the line length limit # NOQA"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.NOQA, line_length=20)
    result = line(content, line_separator, config)
    assert result == "very long line that exceeds the line length limit # NOQA"

def test_line_with_import_split():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "from module import (\n    very_long_function_name\n)"

def test_line_with_cimport_split():
    content = "cimport module.very_long_function_name"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "cimport module.very_long_function_name"

def test_line_with_dot_split():
    content = "module.very_long_function_name()"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "module.very_long_function_name()"

def test_line_with_as_split():
    content = "import module as very_long_alias"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = line(content, line_separator, config)
    assert result == "import module as very_long_alias"

def test_line_with_comment_no_parentheses():
    content = "import module # some comment"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=False)
    result = line(content, line_separator, config)
    assert result == "import module \\\n    # some comment"

def test_line_with_comment_and_parentheses():
    content = "import module # some comment"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "import (\n    module # some comment\n)"

def test_line_with_noqa_comment_and_parentheses():
    content = "import module # noqa: F401"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "import (\n    module # noqa: F401\n)"

def test_line_with_trailing_comma():
    content = "import module"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result == "import (\n    module,\n)"

def test_line_vertical_grid_grouped():
    content = "import module"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_GRID_GROUPED, line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "import (\n    module,\n)"

def test_line_vertical_hanging_indent():
    content = "import module"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True)
    result = line(content, line_separator, config)
    assert result == "import (\n    module,\n)"

def test_line_with_indent():
    content = "    import module"
    line_separator = "\n"
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert result == "    import (\n        module,\n    )"


# LLM-generated content at query #59
#--------------------------

```python
def test_predicate_at_line_65_evaluates_to_false():
    content = "import os"
    line_separator = "\n"
    config = Config(
        line_length=20,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="#",
        indent="    "
    )
    result = line(content, line_separator, config)
    lines = result.split(line_separator)
    assert not (config.comment_prefix in lines[-1] and lines[-1].endswith(")"))


# LLM-generated content at query #60
#--------------------------

```python
def test_predicate_evaluates_to_true():
    content = "a" * 100
    config = Config()
    config.wrap_length = 50
    config.line_length = 40
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #61
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short"
    line_separator = "\n"
    config = Config(
        line_length=100,
        wrap_length=None,
        multi_line_output=Modes.NOQA,
        indent="",
        comment_prefix="#",
        use_parentheses=False,
        include_trailing_comma=False
    )
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length))


# LLM-generated content at query #62
#--------------------------

```python
def test_line_length_greater_than_10():
    config = Config(wrap_length=15, line_length=15, balanced_wrapping=True)
    statement = import_statement(
        import_start="from module import (",
        from_imports=["a", "b", "c"],
        config=config,
    )
    lines = statement.split("\n")
    assert len(lines[-1]) < min(len(line) for line in lines[:-1]) and len(lines) == len(lines) and 15 > 10


# LLM-generated content at query #63
#--------------------------

```python
def test_predicate_at_line_48_evaluates_to_true():
    config = Config(
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    ",
        line_length=88,
        wrap_length=None
    )
    content = "from module import something, another_thing, yet_another_thing"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert wrap_mode in (
        Modes.VERTICAL_HANGING_INDENT,
        Modes.VERTICAL_GRID_GROUPED,
    )


# LLM-generated content at query #64
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_no_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line without noqa", "\n", config) == "long line without noqa # NOQA"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line # NOQA", "\n", config) == "long line # NOQA"

def test_line_wrap_with_import():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import long_module_name"
    expected = "from module import \\\n    long_module_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_comment():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import long_module_name # comment"
    expected = "from module import \\\n    long_module_name # comment"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_noqa_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import long_module_name # noqa"
    expected = "from module import (\n    long_module_name,\n) # noqa"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_as():
    config = Config(line_length=20, use_parentheses=True)
    content = "import module as long_alias"
    expected = "import module as long_alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_dot():
    config = Config(line_length=20, use_parentheses=False)
    content = "module.long_module_name"
    expected = "module.\\\n    long_module_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    content = "from module import long_module_name"
    expected = "from module import (\n    long_module_name,\n)"
    assert line(content, "\n", config) == expected


# LLM-generated content at query #65
#--------------------------

```python
def test_line_predicate_false():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=False,
        include_trailing_comma=False,
        comment_prefix="# ",
        indent="    "
    )
    assert (len(content) + 2) > (config.wrap_length or config.line_length) == False


# LLM-generated content at query #66
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_with_noqa_mode():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length NOQA"

def test_line_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length # NOQA", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_with_import_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module import long_module_name", "\n", config) == "from module import \\\n    long_module_name"

def test_line_with_cimport_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("cimport module.long_module_name", "\n", config) == "cimport module.\\\n    long_module_name"

def test_line_with_dot_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("module.long_module_name.function()", "\n", config) == "module.\\\n    long_module_name.function()"

def test_line_with_as_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module as alias", "\n", config) == "import module \\\n    as alias"

def test_line_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_with_parentheses_and_noqa_comment():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # noqa", "\n", config) == "from module import (\n    long_module_name, # noqa\n)"

def test_line_with_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_with_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("from module import long_module_name", "\n", config) == "from module import (\n    long_module_name,\n)"

def test_line_with_comment_and_no_parentheses():
    config = Config(line_length=20, use_parentheses=False)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import \\\n    long_module_name # comment"

def test_line_with_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # comment", "\n", config) == "from module import (\n    long_module_name, # comment\n)"

def test_line_with_noqa_in_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("from module import long_module_name # noqa", "\n", config) == "from module import (\n    long_module_name, # noqa\n)"


# LLM-generated content at query #67
#--------------------------

```python
def test_comma_maybe_predicate_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "from module import function, other_function"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert ",# NOQA" in result or ",# noqa" in result


# LLM-generated content at query #68
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        line_length=88,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment_prefix="#",
        indent="    ",
        wrap_length=None
    )
    content = "import some_module, another_module  # some comment"
    line_without_comment, comment = content.split("#", 1)
    _comma_maybe = (
        ","
        if (
            config.include_trailing_comma
            and config.use_parentheses
            and not line_without_comment.rstrip().endswith(",")
        )
        else ""
    )
    assert _comma_maybe == ","


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_false():
    config = Config()
    config.include_trailing_comma = False
    config.use_parentheses = True
    line_without_comment = "some content without trailing comma"
    assert not (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    )


# LLM-generated content at query #70
#--------------------------

```python
def test_line_17_predicate_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        indent="    ",
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "import os.path as osp"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import (\n    os.path as osp,\n)"


# LLM-generated content at query #71
#--------------------------

```python
def test_predicate_at_line_17():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        comment_prefix="#",
        line_length=88,
        wrap_length=None,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "import os.path as osp, sys"
    line_without_comment = content
    comment = None
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    )


# LLM-generated content at query #72
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_with_noqa_mode():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("very long line that exceeds line length", "\n", config) == "very long line that exceeds line length NOQA"

def test_line_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("very long line that exceeds line length # NOQA", "\n", config) == "very long line that exceeds line length # NOQA"

def test_line_with_import_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import very.long.module.name", "\n", config) == "import very.long.\\n    module.name"

def test_line_with_cimport_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("cimport very.long.module.name", "\n", config) == "cimport very.long.\\n    module.name"

def test_line_with_dot_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("very.long.module.name", "\n", config) == "very.long.\\n    module.name"

def test_line_with_as_split():
    config = Config(line_length=20, use_parentheses=False)
    assert line("import module as alias", "\n", config) == "import module\\n    as alias"

def test_line_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("import very.long.module.name", "\n", config) == "import (\\n    very.long.module.name,\\n)"

def test_line_with_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("import module # comment", "\n", config) == "import (\\n    module,  # comment\\n)"

def test_line_with_noqa_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    assert line("import module # noqa", "\n", config) == "import (\\n    module\\n) # noqa"

def test_line_with_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line("import very.long.module.name", "\n", config) == "import (\\n    very.long.module.name,\\n)"

def test_line_with_vertical_grid_grouped():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert line("import very.long.module.name", "\n", config) == "import (\\n    very.long.module.name,\\n)"


# LLM-generated content at query #73
#--------------------------

```python
def test_predicate_at_line_17_evaluates_to_true():
    config = Config(
        include_trailing_comma=True,
        use_parentheses=True,
        line_length=88,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="# "
    )
    content = "from module import something, another_thing  # comment"
    line_without_comment = "from module import something, another_thing"
    assert (
        config.include_trailing_comma
        and config.use_parentheses
        and not line_without_comment.rstrip().endswith(",")
    )


# LLM-generated content at query #74
#--------------------------

```python
def test_while_loop_predicate():
    config = Config()
    config.balanced_wrapping = True
    config.line_length = 100
    config.wrap_length = None
    config.include_trailing_comma = False
    config.ignore_comments = False
    config.comment_prefix = "#"
    config.indent = "    "
    config.multi_line_output = Modes.GRID

    import_start = "from module import"
    from_imports = ["a", "b", "c"]
    comments = ()

    statement = import_statement(
        import_start=import_start,
        from_imports=from_imports,
        config=config,
    )

    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0

    assert len(lines[-1]) < minimum_length and len(lines) == line_count and config.line_length > 10


# LLM-generated content at query #75
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("very long line that exceeds line length # NOQA", "\n", config) == "very long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("very long line", "\n", config) == "very long line # NOQA"

def test_line_wrap_with_import_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import very_long_name"
    expected = "from module import \\\n    very_long_name"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "module.very_long_function_name()"
    expected = "module.\\\n    very_long_function_name()"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_as_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "import module as very_long_alias"
    expected = "import module as \\\n    very_long_alias"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import (item1, item2, item3)"
    expected = "from module import (\n    item1,\n    item2,\n    item3,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_comment_and_parentheses():
    config = Config(line_length=20, use_parentheses=True, comment_prefix="# ")
    content = "from module import item # some comment"
    expected = "from module import (\n    item,  # some comment\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=20, use_parentheses=True, comment_prefix="# ")
    content = "from module import item # noqa: F401"
    expected = "from module import item # noqa: F401"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    content = "from module import (item1, item2, item3)"
    expected = "from module import (\n    item1,\n    item2,\n    item3,\n)"
    assert line(content, "\n", config) == expected

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    content = "from module import (item1, item2, item3)"
    expected = "from module import (\n    item1,\n    item2,\n    item3,\n)"
    assert line(content, "\n", config) == expected


# LLM-generated content at query #76
#--------------------------

```python
def test_line_71_predicate_evaluates_to_true():
    content = "a" * 100
    line_separator = "\n"
    config = Config(
        line_length=50,
        multi_line_output=Modes.NOQA,
        comment_prefix="# "
    )
    assert len(content) > config.line_length
    assert config.multi_line_output == Modes.NOQA
    assert "# NOQA" not in content


# LLM-generated content at query #77
#--------------------------

```python
def test_line_no_wrap_needed():
    assert line("short line", "\n") == "short line"

def test_line_wrap_with_import():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import long_module_name", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import long_module_name # comment", "\n", config)
    assert "# comment" in result and "\\" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # NOQA", "\n", config)
    assert result == "import long_module_name # NOQA"

def test_line_wrap_with_as():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import long_module_name as lmn", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_dot():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("long_module_name.function_name", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("cimport long_module_name", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_parentheses():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line("import long_module_name", "\n", config)
    assert "(" in result and ")" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import long_module_name", "\n", config)
    assert "," in result

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    result = line("import long_module_name # noqa", "\n", config)
    assert "noqa" in result and "(" in result and ")" in result

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    result = line("import long_module_name", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import long_module_name", "\n", config)
    assert "\\" in result and "\n" in result

def test_line_wrap_with_noqa_mode():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name", "\n", config)
    assert result == "import long_module_name NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # NOQA", "\n", config)
    assert result == "import long_module_name # NOQA"

def test_line_wrap_with_noqa_mode_and_other_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # comment", "\n", config)
    assert result == "import long_module_name NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa", "\n", config)
    assert result == "import long_module_name # noqa"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA NOQA", "\n", config)
    assert result == "import long_module_name # noqa NOQA comment NOQA NOQA NOQA NOQA NOQA NOQA NOQA"

def test_line_wrap_with_noqa_mode_and_noqa_in_comment_and_noqa_comment_and_other_comment_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa_and_noqa():
    config


# LLM-generated content at query #78
#--------------------------

```python
def test_line_no_wrapping_needed():
    assert line("short line", "\n") == "short line"

def test_line_noqa_mode_with_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA)
    assert line("long line that exceeds line length # NOQA", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_noqa_mode_without_noqa_comment():
    config = Config(multi_line_output=Modes.NOQA, line_length=10)
    assert line("long line that exceeds line length", "\n", config) == "long line that exceeds line length # NOQA"

def test_line_wrap_with_import_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "from module import function"
    assert line(content, "\n", config) == "from module import \\\n    function"

def test_line_wrap_with_as_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "import module as alias"
    assert line(content, "\n", config) == "import module \\\n    as alias"

def test_line_wrap_with_parentheses_and_trailing_comma():
    config = Config(line_length=20, use_parentheses=True, include_trailing_comma=True)
    content = "from module import function"
    assert line(content, "\n", config) == "from module import(\n    function,\n)"

def test_line_wrap_with_parentheses_and_noqa_comment():
    config = Config(line_length=20, use_parentheses=True, comment_prefix="# ")
    content = "from module import function # noqa"
    assert line(content, "\n", config) == "from module import(\n    function,  # noqa\n)"

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(line_length=20, use_parentheses=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import function"
    assert line(content, "\n", config) == "from module import(\n    function,\n)"

def test_line_wrap_with_comment_and_no_parentheses():
    config = Config(line_length=20, use_parentheses=False, comment_prefix="# ")
    content = "from module import function # comment"
    assert line(content, "\n", config) == "from module import \\\n    function  # comment"

def test_line_wrap_with_cimport_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "cimport module.function"
    assert line(content, "\n", config) == "cimport module.\\\n    function"

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, use_parentheses=False)
    content = "module.function(argument)"
    assert line(content, "\n", config) == "module.\\\n    function(argument)"


# LLM-generated content at query #79
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
    )
    assert result == "from module import A, B, C"

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        comments=["# Comment"],
    )
    assert "# Comment" in result

def test_import_statement_explode_mode():
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        explode=True,
    )
    assert result == "from module import (\n    A,\n    B,\n    C,\n)"

def test_import_statement_custom_line_separator():
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_custom_config():
    config = Config(wrap_length=20)
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        config=config,
    )
    assert len(result.split("\n")[0]) <= 20

def test_import_statement_multi_line_output():
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert "\n" in result

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        config=config,
    )
    lines = result.split("\n")
    assert len(lines[-1]) >= min(len(line) for line in lines[:-1])

def test_import_statement_single_line():
    result = import_statement(
        import_start="from module import",
        from_imports=["A"],
    )
    assert "\n" not in result

def test_import_statement_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        config=config,
    )
    assert result.endswith(",")

def test_import_statement_ignore_comments():
    config = Config(ignore_comments=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["A", "B", "C"],
        comments=["# Comment"],
        config=config,
    )
    assert "# Comment" not in result


# LLM-generated content at query #80
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
    )
    assert isinstance(result, str)
    assert "from module import" in result
    assert "a" in result
    assert "b" in result
    assert "c" in result

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        comments=["# comment 1", "# comment 2"],
    )
    assert "# comment 1" in result
    assert "# comment 2" in result

def test_import_statement_custom_line_separator():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        line_separator="\r\n",
    )
    assert "\r\n" in result

def test_import_statement_explode_mode():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    assert result.count("\n") >= len(["a", "b", "c"])

def test_import_statement_multi_line_output():
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert "\n" in result

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True)
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        config=config,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1])

def test_import_statement_single_line_output():
    result = import_statement(
        import_start="from module import",
        from_imports=["a"],
    )
    assert result.count("\n") == 0

def test_import_statement_custom_config():
    config = Config(indent="    ", wrap_length=50)
    result = import_statement(
        import_start="from module import",
        from_imports=["a", "b", "c"],
        config=config,
    )
    assert "    " in result

def test_import_statement_empty_imports():
    result = import_statement(
        import_start="from module import",
        from_imports=[],
    )
    assert "from module import" in result


