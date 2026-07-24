####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=3))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import(" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, multi_line_output=3, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("import os  # comment", "\n", config)
    assert result == "import os  # comment"

def test_line_wrap_noqa_mode():
    config = Config(line_length=10, multi_line_output=5, comment_prefix="  # ")
    result = line("import os", "\n", config)
    assert result == "import os  # NOQA"

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "very_long_module_name as" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("os.path.join", "\n", config)
    assert "os." in result
    assert "path.join" in result

def test_line_wrap_with_cimport_split():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("cimport numpy as np", "\n", config)
    assert "cimport numpy as" in result
    assert "np" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import function", "\n", config)
    assert result.endswith(",")

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=30, multi_line_output=3, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("import os  # noqa", "\n", config)
    assert "noqa" in result

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=4, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("from module import function", "\n", config)
    assert "from module import(" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=5, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("from module import function", "\n", config)
    assert "from module import(" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=False, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("from module import function", "\n", config)
    assert "\\" in result

def test_line_empty_content():
    result = line("", "\n", Config(line_length=10, multi_line_output=3))
    assert result == ""

def test_line_exact_line_length():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=False)
    result = line("import os", "\n", config)
    assert result == "import os"


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_15_true():
    config = Config()
    config.use_parentheses = False
    comment = "some comment"
    result = comment and not (config.use_parentheses and "noqa" in comment)
    assert result == True


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_71_true():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "a" * 81
    result = line(content, "\n", config)
    assert result == content + config.comment_prefix + " NOQA"


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_56_evaluates_to_true():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    config.line_length = 80
    config.wrap_length = None
    config.indent = "    "
    content = "from very_long_module_name import very_long_submodule_name as very_long_alias_name"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "noqa" in result


# LLM-generated content at query #5
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("very_long_module_name.very_long_submodule", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something" in result
    assert "# some comment" in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import verylongmodule", "\n", config)
    assert result == "import verylongmodule  # NOQA"

def test_line_wrap_noqa_mode_with_existing_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_with_comment_and_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import something  # comment", "\n", config)
    assert "from module import (" in result
    assert "something  # comment" in result
    assert result.endswith(")")

def test_line_wrap_no_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=None)
    result = line("from module import something", "\n", config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_wrap_with_noqa_in_comment_and_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "something" in result
    assert result.endswith(")")

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from libc.math cimport sin", "\n", config)
    assert "from libc.math cimport (" in result
    assert "sin" in result


# LLM-generated content at query #6
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result or "import very_long_module_name as \\" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=25, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("very_long_module_name.very_long_submodule", "\n", config)
    assert "very_long_module_name.(" in result or "very_long_module_name.\\" in result

def test_line_with_comment_and_wrap():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something" in result
    assert "# some comment" in result

def test_line_noqa_mode_with_long_line():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import os", "\n", config)
    assert result == "import os  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import os  # NOQA", "\n", config)
    assert result == "import os  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert result.endswith(",") or "something," in result

def test_line_wrap_with_comment_and_noqa_in_comment():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import something  # noqa", "\n", config)
    assert "# noqa" in result
    assert ")" in result

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from module import something", "\n", config)
    assert "\\" in result


# LLM-generated content at query #7
#--------------------------

def test_include_trailing_comma_with_parentheses_and_no_trailing_comma_in_line_without_comment():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    config.comment_prefix = "  #"
    content = "from module import something  # comment"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "," in result


# LLM-generated content at query #8
#--------------------------

def test_import_statement_explode_mode():
    config = Config()
    result = import_statement("from module", ["item1", "item2"], explode=True)
    expected = "from module import (\n    item1,\n    item2,\n)"
    assert result == expected

def test_import_statement_default_formatter():
    config = Config(multi_line_output=Modes.GRID, wrap_length=50, line_length=50)
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert result == "from module import item1, item2"

def test_import_statement_with_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20)
    comments = ["comment1", "comment2"]
    result = import_statement("from module", ["item1", "item2"], comments=comments, config=config)
    lines = result.split("\n")
    assert any("comment1" in line for line in lines)
    assert any("comment2" in line for line in lines)

def test_import_statement_balanced_wrapping():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, balanced_wrapping=True)
    imports = ["item" + str(i) for i in range(10)]
    result = import_statement("from module", imports, config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_single_line_wrap():
    config = Config(multi_line_output=Modes.GRID, wrap_length=10)
    result = import_statement("from module", ["verylongimportname"], config=config)
    assert result.count("\n") > 0

def test_import_statement_trailing_comma():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, wrap_length=20)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.strip().endswith(",")

def test_import_statement_no_trailing_comma():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=False, wrap_length=20)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert not result.strip().endswith(",")

def test_import_statement_custom_line_separator():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20)
    result = import_statement("from module", ["item1", "item2"], line_separator="\r\n", config=config)
    assert "\r\n" in result

def test_import_statement_remove_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, ignore_comments=True)
    comments = ["comment1", "comment2"]
    result = import_statement("from module", ["item1", "item2"], comments=comments, config=config)
    assert "comment1" not in result
    assert "comment2" not in result

def test_import_statement_indent():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, indent="    ")
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert "    " in result


# LLM-generated content at query #9
#--------------------------

def test_import_statement_explode_mode():
    result = import_statement("from module", ["item1", "item2"], explode=True)
    expected = "from module import (\n    item1,\n    item2,\n)"
    assert result == expected

def test_import_statement_default_formatter():
    config = Config(multi_line_output=Modes.GRID, line_length=80, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert "from module import item1, item2" in result

def test_import_statement_with_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=40, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"], config=config)
    assert "comment1" in result and "comment2" in result

def test_import_statement_balanced_wrapping():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=30, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False, balanced_wrapping=True)
    result = import_statement("from module", ["very_long_item_name1", "very_long_item_name2", "item3"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_single_line_wrap():
    config = Config(multi_line_output=Modes.GRID, line_length=20, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.count("\n") == 0

def test_import_statement_custom_line_separator():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=40, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2"], line_separator="\r\n", config=config)
    assert "\r\n" in result

def test_import_statement_remove_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=40, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=True)
    result = import_statement("from module", ["item1", "item2"], comments=["comment1"], config=config)
    assert "comment1" not in result

def test_import_statement_no_wrap_needed():
    config = Config(multi_line_output=Modes.GRID, line_length=100, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert result == "from module import item1, item2"

def test_import_statement_wrap_length_override():
    config = Config(multi_line_output=Modes.GRID, line_length=100, wrap_length=20, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.count("\n") > 0

def test_import_statement_vertical_hanging_indent():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=30, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.startswith("from module import (")


# LLM-generated content at query #10
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import(" in result
    assert "    very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "    very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "    very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import(" in result
    assert "    something,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import(  # noqa" in result
    assert "    something" in result

def test_line_noqa_mode_with_long_line():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert "from module import\\" in result
    assert "    something" in result

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something  # comment", "\n", config)
    assert "from module import(" in result
    assert "    something,  # comment" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import(" in result
    assert "    something," in result


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_15_true():
    config = Config()
    config.use_parentheses = False
    comment = "some comment"
    result = not (config.use_parentheses and "noqa" in comment)
    assert result == True


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_56_true():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    config.indent = "    "
    config.wrap_length = None
    config.line_length = 80
    content = "from module import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    comment = "noqa"
    line_without_comment = content
    splitter = "import "
    line_parts = re.split(r"\b" + re.escape(splitter) + r"\b", line_without_comment)
    line_parts[-1] = f"{line_parts[-1].strip()}{config.comment_prefix}{comment}"
    next_line = []
    while (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts:
        next_line.append(line_parts.pop())
        content = splitter.join(line_parts)
    if not content:
        content = next_line.pop()
    cont_line = _wrap_line(config.indent + splitter.join(next_line).lstrip(), line_separator, config)
    noqa_comment = ""
    if comment and "noqa" in comment:
        noqa_comment = f"{config.comment_prefix}{comment}"
        cont_line = cont_line.rstrip()
        _comma = "," if config.include_trailing_comma else ""
    output = f"{content}{splitter}({noqa_comment}{line_separator}{cont_line}{_comma})"
    lines = output.split(line_separator)
    assert config.comment_prefix in lines[-1]
    assert lines[-1].endswith(")")


# LLM-generated content at query #13
#--------------------------

def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=100, multi_line_output=Modes.GRID)
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    expected_start = "from very_long_module_name import ("
    expected_end = "very_long_function_name,"
    assert result.startswith(expected_start)
    assert expected_end in result

def test_line_wrap_with_as_split():
    content = "import very_long_module_name as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "very_long_module_name as" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    content = "very_long_module_name.very_long_submodule.very_long_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "very_long_module_name." in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    content = "from module import something  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "# some comment" in result

def test_line_wrap_with_noqa_comment():
    content = "from module import something  # noqa"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "# noqa" in result
    assert result.endswith("# noqa)")

def test_line_wrap_mode_noqa():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="# ")
    result = line(content, line_separator, config)
    assert result == "from module import something#  NOQA"

def test_line_wrap_mode_noqa_with_existing_noqa():
    content = "from module import something  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="# ")
    result = line(content, line_separator, config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_without_parentheses():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=False, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "\\" in result
    assert "from very_long_module_name import" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_trailing_comma():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert result.endswith(",")

def test_line_wrap_without_trailing_comma():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="# ", include_trailing_comma=False)
    result = line(content, line_separator, config)
    assert not result.endswith(",")


# LLM-generated content at query #14
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=True, indent="    ")
    result = line("from verylongmodule import verylongname", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname)"

def test_line_wrap_with_as_split():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=True, indent="    ")
    result = line("import verylongmodule as vl", "\n", config)
    assert result == "import verylongmodule as(\n    vl)"

def test_line_wrap_with_dot_split():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=True, indent="    ")
    result = line("verylongmodule.verylongname", "\n", config)
    assert result == "verylongmodule.(\n    verylongname)"

def test_line_wrap_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line("from verylongmodule import verylongname  # comment", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname  # comment)"

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line("from verylongmodule import verylongname  # noqa", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname  # noqa)"

def test_line_wrap_without_parentheses():
    config = Config(line_length=10, multi_line_output=Modes.GRID, wrap_length=10, use_parentheses=False, indent="    ")
    result = line("from verylongmodule import verylongname", "\n", config)
    assert result == "from verylongmodule import\\\n    verylongname"

def test_line_wrap_noqa_mode():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("verylongmoduleverylongname", "\n", config)
    assert result == "verylongmoduleverylongname  # NOQA"

def test_line_wrap_noqa_mode_with_existing_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("verylongmoduleverylongname  # NOQA", "\n", config)
    assert result == "verylongmoduleverylongname  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=10, use_parentheses=True, indent="    ", include_trailing_comma=True)
    result = line("from verylongmodule import verylongname", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname,\n)"

def test_line_wrap_with_comment_and_trailing_comma():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=10, use_parentheses=True, indent="    ", include_trailing_comma=True, comment_prefix="  # ")
    result = line("from verylongmodule import verylongname  # comment", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname,  # comment\n)"

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_GRID_GROUPED, wrap_length=10, use_parentheses=True, indent="    ", include_trailing_comma=True)
    result = line("from verylongmodule import verylongname", "\n", config)
    assert result == "from verylongmodule import(\n    verylongname,\n)"


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_71_true():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "a" * 81
    result = line(content, "\n", config)
    assert result == content + config.comment_prefix + " NOQA"


# LLM-generated content at query #16
#--------------------------

def test_balanced_wrapping_condition_true():
    import_start = "from module import"
    from_imports = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="# ", include_trailing_comma=False, ignore_comments=False, multi_line_output=Modes.GRID)
    multi_line_output = None
    explode = False
    line_separator = "\n"
    comments = []
    statement = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    lines = statement.split(line_separator)
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0
    condition = len(lines[-1]) < minimum_length and len(lines) == line_count and config.wrap_length > 10
    assert condition == True


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_46_true():
    config = Config()
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    config.use_parentheses = True
    config.wrap_length = None
    config.line_length = 100
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.indent = "    "
    line_separator = "\n"
    content = "very_long_module_name"
    splitter = "import "
    comment = None
    _comma = "," if config.include_trailing_comma and not comment else ""
    assert _comma == ","


# LLM-generated content at query #18
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name," in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("very_long_module_name.very_long_attribute", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_attribute" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "something," in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20)
    result = line("from module import something", "\n", config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_with_trailing_comma_no_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "something," in result

def test_line_wrap_without_trailing_comma_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("from module import something  # comment", "\n", config)
    assert "something  # comment" in result
    assert not result.endswith(",")

def test_line_wrap_empty_content():
    result = line("", "\n", Config(line_length=20, multi_line_output=Modes.GRID))
    assert result == ""


# LLM-generated content at query #19
#--------------------------

```python
def test_balanced_wrapping_condition_true():
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="  #", ignore_comments=False, include_trailing_comma=False, multi_line_output=Modes.GRID)
    import_start = "from module import"
    from_imports = ["very_long_import_name_that_will_cause_wrapping", "another_import", "third_import"]
    comments = []
    line_separator = "\n"
    multi_line_output = None
    explode = False
    result = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    lines = result.split(line_separator)
    assert len(lines) > 1
    last_line_length = len(lines[-1])
    other_lines_lengths = [len(line) for line in lines[:-1]]
    minimum_other_length = min(other_lines_lengths) if other_lines_lengths else 0
    assert last_line_length < minimum_other_length


# LLM-generated content at query #20
#--------------------------

def test_import_statement_explode_mode():
    config = Config()
    result = import_statement("from module import", ["a", "b", "c"], explode=True)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_single_line():
    config = Config()
    config.multi_line_output = Modes.GRID
    config.line_length = 100
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    expected = "from module import a, b, c"
    assert result == expected

def test_import_statement_multi_line_grid():
    config = Config()
    config.multi_line_output = Modes.GRID
    config.line_length = 20
    result = import_statement("from module import", ["a", "b", "c", "d", "e"], config=config)
    expected = "from module import a, b, c,\n              d, e"
    assert result == expected

def test_import_statement_with_comments():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 30
    comments = ["comment1", "comment2"]
    result = import_statement("from module import", ["a", "b"], comments=comments, config=config)
    expected = "from module import (  # comment1\n    a,  # comment2\n    b,\n)"
    assert result == expected

def test_import_statement_balanced_wrapping():
    config = Config()
    config.multi_line_output = Modes.GRID
    config.line_length = 25
    config.balanced_wrapping = True
    result = import_statement("from module import", ["a", "b", "c", "d", "e", "f"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 25 for line in lines)

def test_import_statement_include_trailing_comma():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 20
    config.include_trailing_comma = True
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_custom_indent():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 20
    config.indent = "    "
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_line_separator():
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 20
    result = import_statement("from module import", ["a", "b", "c"], line_separator="\r\n", config=config)
    expected = "from module import (\r\n    a,\r\n    b,\r\n    c,\r\n)"
    assert result == expected

def test_import_statement_wrap_length_overrides_line_length():
    config = Config()
    config.multi_line_output = Modes.GRID
    config.line_length = 100
    config.wrap_length = 20
    result = import_statement("from module import", ["a", "b", "c", "d", "e"], config=config)
    expected = "from module import a, b, c,\n              d, e"
    assert result == expected

def test_import_statement_no_wrap_length_uses_line_length():
    config = Config()
    config.multi_line_output = Modes.GRID
    config.line_length = 20
    config.wrap_length = None
    result = import_statement("from module import", ["a", "b", "c", "d", "e"], config=config)
    expected = "from module import a, b, c,\n              d, e"
    assert result == expected


# LLM-generated content at query #21
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  #")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  #")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "something" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_no_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20)
    result = line("from module import something", "\n", config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_short_content():
    result = line("import os", "\n", Config(line_length=10, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something" in result

def test_line_with_backslash_separator():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20)
    result = line("from module import something", "\\", config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_comment_with_noqa_and_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  #")
    result = line("from module import something  # noqa comment", "\n", config)
    assert "from module import (  # noqa comment" in result
    assert "something" in result

def test_line_empty_content():
    result = line("", "\n", Config(line_length=10, multi_line_output=Modes.GRID))
    assert result == ""


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_17_true():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    line_without_comment = "some_import_statement"
    comment = "some comment"
    result = (
        ","
        if (
            config.include_trailing_comma
            and config.use_parentheses
            and not line_without_comment.rstrip().endswith(",")
        )
        else ""
    )
    assert result == ","


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.multi_line_output = 1
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    line_separator = "\n"
    line_parts = ["part1", "part2"]
    result = (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts
    assert result == False


# LLM-generated content at query #24
#--------------------------

```python
def test_balanced_wrapping_condition_true():
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="# ", ignore_comments=False, include_trailing_comma=False, multi_line_output=Modes.GRID)
    import_start = "from module import"
    from_imports = ["very_long_import_name_that_will_cause_wrapping", "another_import", "third_import", "fourth_import"]
    statement = import_statement(import_start, from_imports, config=config, multi_line_output=Modes.GRID)
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if line_count > 1 else 0
    condition_result = len(lines[-1]) < minimum_length and len(lines) == line_count and config.wrap_length > 10
    assert condition_result == True


# LLM-generated content at query #25
#--------------------------

```python
def test_formatter_from_string_returns_grid_for_unknown_name():
    result = formatter_from_string("UNKNOWN_FORMATTER")
    expected = grid
    assert result == expected


# LLM-generated content at query #26
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert result == "from very_long_module_name import(\n    very_long_function_name,\n)"

def test_line_wrap_with_as_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert result == "import very_long_module_name as vlm"

def test_line_wrap_with_dot_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("very_long_module_name.very_long_submodule", "\n", config)
    assert result == "very_long_module_name.very_long_submodule"

def test_line_wrap_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("import very_long_module_name  # some comment", "\n", config)
    assert result == "import very_long_module_name  # some comment"

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ", use_parentheses=False, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name#  NOQA"

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ", use_parentheses=False, include_trailing_comma=False)
    result = line("import very_long_module_name  # NOQA", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=False, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert result == "from very_long_module_name import\\\n    very_long_function_name"

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert result == "from very_long_module_name import(\n    very_long_function_name,\n)"

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("import very_long_module_name  # comment", "\n", config)
    assert result == "import very_long_module_name  # comment"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=80, multi_line_output=3))
    assert result == "import os"

def test_line_wrap_on_import():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "# some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (" in result
    assert "# noqa" in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=20, multi_line_output=5, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_with_as_keyword():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result or "import very_long_module_name as vlm" in result

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("module.submodule.very_long_attribute", "\n", config)
    assert "module.submodule.(" in result
    assert "very_long_attribute" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("cimport very_long_module_name", "\n", config)
    assert "cimport very_long_module_name" in result

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=4, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert "from module import (" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=5, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert "from module import (" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=False, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert "\\" in result

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result.endswith(",") or "something," in result

def test_line_wrap_without_trailing_comma():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=False, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert not result.endswith(",")

def test_line_wrap_with_comment_and_noqa():
    config = Config(line_length=20, multi_line_output=3, wrap_length=20, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something  # comment with noqa", "\n", config)
    assert "# comment with noqa" in result

def test_line_wrap_empty_content():
    result = line("", "\n", Config(line_length=20, multi_line_output=3))
    assert result == ""

def test_line_wrap_exact_line_length():
    config = Config(line_length=30, multi_line_output=3, wrap_length=30, use_parentheses=True, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something"


# LLM-generated content at query #2
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import(" in result
    assert "    very_long_function_name," in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "    very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "    very_long_submodule.very_long_function)" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import(" in result
    assert "    something,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import(  # noqa" in result
    assert "    something," in result

def test_line_wrap_mode_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_mode_noqa_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, wrap_length=20, indent="    ", comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert "from module import\\" in result
    assert "    something" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import(" in result
    assert "    something," in result

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import something  # comment", "\n", config)
    assert "from module import(" in result
    assert "    something,  # comment" in result

def test_line_wrap_without_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=20, indent="    ", comment_prefix="  #", include_trailing_comma=False)
    result = line("from module import something", "\n", config)
    assert "from module import(" in result
    assert "    something" in result
    assert result.endswith(")")

def test_line_wrap_complex_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, wrap_length=30, indent="    ", comment_prefix="  #", include_trailing_comma=True)
    result = line("import very_long_module_name.submodule as alias", "\n", config)
    assert "import very_long_module_name.(" in result
    assert "    submodule as alias" in result


# LLM-generated content at query #3
#--------------------------

def test_predicate_at_line_30_evaluates_to_false():
    config = Config()
    config.wrap_length = None
    config.line_length = 100
    content = "a" * 50
    line_parts = ["part1", "part2"]
    result = (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts
    assert result == False


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_17_false():
    config = Config()
    config.include_trailing_comma = False
    config.use_parentheses = True
    line_without_comment = "something"
    comment = "some comment"
    result = config.include_trailing_comma and config.use_parentheses and not line_without_comment.rstrip().endswith(",")
    assert result == False


# LLM-generated content at query #5
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=100, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import(" in result
    assert "    very_long_function_name," in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "    very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "    very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import(" in result
    assert "    something,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import(  # noqa" in result
    assert "    something," in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, wrap_length=20, use_parentheses=False, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import\\" in result
    assert "    something" in result

def test_line_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import something  # comment", "\n", config)
    assert result.endswith(",  # comment)")

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import(" in result
    assert "    something," in result


# LLM-generated content at query #6
#--------------------------

def test_import_statement_explode_mode():
    result = import_statement("from module import", ["a", "b", "c"], explode=True)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_default_formatter():
    config = Config(multi_line_output=Modes.GRID, wrap_length=None, line_length=80, indent="    ", include_trailing_comma=False, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c"], config=config)
    assert result == "from module import a, b, c"

def test_import_statement_with_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=None, line_length=20, indent="    ", include_trailing_comma=True, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c"], comments=["comment1", "comment2", "comment3"], config=config)
    expected = "from module import (\n    a,  # comment1\n    b,  # comment2\n    c,  # comment3\n)"
    assert result == expected

def test_import_statement_balanced_wrapping():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=None, line_length=30, indent="    ", include_trailing_comma=True, ignore_comments=False, comment_prefix="  #", balanced_wrapping=True)
    result = import_statement("from module import", ["very_long_import_name_a", "very_long_import_name_b", "very_long_import_name_c"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_single_line_wrap():
    config = Config(multi_line_output=Modes.GRID, wrap_length=10, line_length=10, indent="    ", include_trailing_comma=False, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c", "d", "e"], config=config)
    assert result.count("\n") >= 1

def test_import_statement_custom_line_separator():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=None, line_length=20, indent="    ", include_trailing_comma=True, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c"], line_separator="\r\n", config=config)
    expected = "from module import (\r\n    a,\r\n    b,\r\n    c,\r\n)"
    assert result == expected

def test_import_statement_no_imports():
    config = Config(multi_line_output=Modes.GRID, wrap_length=None, line_length=80, indent="    ", include_trailing_comma=False, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", [], config=config)
    assert result == "from module import "

def test_import_statement_remove_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=None, line_length=20, indent="    ", include_trailing_comma=True, ignore_comments=True, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c"], comments=["comment1", "comment2", "comment3"], config=config)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_multi_line_output_override():
    config = Config(multi_line_output=Modes.GRID, wrap_length=None, line_length=80, indent="    ", include_trailing_comma=False, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c"], multi_line_output=Modes.VERTICAL_HANGING_INDENT, config=config)
    expected = "from module import (\n    a,\n    b,\n    c,\n)"
    assert result == expected

def test_import_statement_wrap_length_specified():
    config = Config(multi_line_output=Modes.GRID, wrap_length=30, line_length=80, indent="    ", include_trailing_comma=False, ignore_comments=False, comment_prefix="  #", balanced_wrapping=False)
    result = import_statement("from module import", ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], config=config)
    assert len(result.split("\n")) > 1


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_71_false():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "some_very_long_import_statement_that_exceeds_line_length # NOQA"
    result = line(content, "\n", config)
    assert not (len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content)


# LLM-generated content at query #8
#--------------------------

def test_predicate_at_line_43_true():
    config = Config()
    config.use_parentheses = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 80
    config.wrap_length = None
    config.indent = "    "
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    content = "very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    splitter = "as "
    cont_line = "short_name"
    output = f"{content}{splitter}{cont_line.lstrip()}"
    assert "as " in output


# LLM-generated content at query #9
#--------------------------

def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=100, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_wrap_with_import_splitter():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    expected_start = "from very_long_module_name import ("
    assert result.startswith(expected_start)
    assert "very_long_function_name" in result

def test_line_wrap_with_dot_splitter():
    content = "module.submodule.very_long_attribute_name"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "module.submodule." in result
    assert "very_long_attribute_name" in result

def test_line_wrap_with_as_splitter():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "import very_long_module_name as" in result
    assert "vlm" in result

def test_line_wrap_with_comment():
    content = "import os  # comment"
    line_separator = "\n"
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "import os" in result
    assert "comment" in result

def test_line_wrap_noqa_mode():
    content = "import very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.endswith("  # NOQA")

def test_line_wrap_noqa_present():
    content = "import os  # NOQA"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os  # NOQA"

def test_line_wrap_vertical_hanging_indent():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.startswith("from module import (")
    assert "very_long_function_name" in result

def test_line_wrap_without_parentheses():
    content = "import very_long_module_name"
    line_separator = "\n"
    config = Config(line_length=25, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "\\" in result
    assert "\n" in result

def test_line_wrap_comment_with_noqa_and_parentheses():
    content = "import os  # noqa"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "noqa" in result
    assert result.endswith(")")

def test_line_wrap_include_trailing_comma():
    content = "from module import function"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.endswith(",")

def test_line_wrap_no_trailing_comma_with_comment():
    content = "import os  # comment"
    line_separator = "\n"
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert not result.endswith(",")

def test_line_wrap_cimport_splitter():
    content = "cimport numpy as np"
    line_separator = "\n"
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert "cimport numpy as" in result
    assert "np" in result

def test_line_wrap_starts_with_splitter():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=5, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_wrap_length_override():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=100, wrap_length=30, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.startswith("from module import (")

def test_line_wrap_vertical_grid_grouped():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.startswith("from module import (")
    assert "very_long_function_name" in result


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #11
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  # ")
    result = line("from module import function  # some comment", "\n", config)
    assert "from module import (" in result
    assert "function" in result
    assert "# some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  # ")
    result = line("from module import function  # noqa", "\n", config)
    assert "from module import (" in result
    assert "function" in result
    assert "# noqa" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20)
    result = line("from module import function", "\n", config)
    assert "from module import\\" in result
    assert "function" in result

def test_line_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import function", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import function  # NOQA", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import function", "\n", config)
    assert "from module import (" in result
    assert "function," in result

def test_line_with_comment_and_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import function  # comment", "\n", config)
    assert "from module import (" in result
    assert "function,  # comment" in result


# LLM-generated content at query #12
#--------------------------

def test_predicate_at_line_43_true():
    config = Config()
    config.use_parentheses = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 80
    config.wrap_length = None
    config.indent = "    "
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    content = "from very_long_module_name import very_long_submodule_name"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert config.use_parentheses


# LLM-generated content at query #13
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name," in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function)" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  #", include_trailing_comma=True)
    result = line("from module import function  # some comment", "\n", config)
    assert "from module import (" in result
    assert "function,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, comment_prefix="  #")
    result = line("from module import function  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "function" in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import function", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_wrap_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import function  # NOQA", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20)
    result = line("from module import function", "\n", config)
    assert "from module import \\" in result
    assert "function" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import function", "\n", config)
    assert "from module import (" in result
    assert "function," in result

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True, comment_prefix="  #")
    result = line("from module import function  # comment", "\n", config)
    assert "function,  # comment" in result

def test_line_wrap_without_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False, comment_prefix="  #")
    result = line("from module import function  # comment", "\n", config)
    assert "function  # comment" in result
    assert not result.endswith(",")

def test_line_wrap_content_empty_after_split():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=10)
    result = line("import module", "\n", config)
    assert result == "import module"


# LLM-generated content at query #14
#--------------------------

def test_import_statement_balanced_wrapping_predicate_false():
    import_start = "from module import "
    from_imports = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    config = Config(multi_line_output=Modes.GRID, wrap_length=50, line_length=50, balanced_wrapping=True, indent="    ", comment_prefix="  # ", ignore_comments=False, include_trailing_comma=False)
    statement = import_statement(import_start, from_imports, config=config, multi_line_output=Modes.GRID)
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if line_count > 1 else 0
    last_line_length = len(lines[-1])
    predicate = last_line_length < minimum_length and len(lines) == line_count and config.line_length > 10
    assert predicate == False


# LLM-generated content at query #15
#--------------------------

def test_predicate_at_line_71_evaluates_to_false():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "some_very_long_import_statement_that_exceeds_line_length_by_a_lot"
    result = line(content, "\n", config)
    assert "# NOQA" in content or len(content) <= config.line_length or config.multi_line_output != Modes.NOQA


# LLM-generated content at query #16
#--------------------------

def test_import_statement_explode_mode():
    result = import_statement("from module", ["item1", "item2"], explode=True)
    expected = "from module import (\n    item1,\n    item2,\n)"
    assert result == expected

def test_import_statement_single_line():
    config = Config(line_length=100, multi_line_output=Modes.GRID)
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert result == "from module import item1, item2"

def test_import_statement_multi_line_grid():
    config = Config(line_length=20, multi_line_output=Modes.GRID)
    result = import_statement("from module", ["item1", "item2", "item3", "item4"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_with_comments():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"], config=config)
    assert "# comment1" in result
    assert "# comment2" in result

def test_import_statement_balanced_wrapping():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, balanced_wrapping=True)
    result = import_statement("from module", ["item1", "item2", "item3", "item4", "item5"], config=config)
    lines = result.split("\n")
    lengths = [len(line) for line in lines[:-1]]
    assert max(lengths) - min(lengths) <= 1

def test_import_statement_include_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.endswith(",\n)")

def test_import_statement_custom_line_separator():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from module", ["item1", "item2", "item3"], line_separator="\r\n", config=config)
    assert "\r\n" in result

def test_import_statement_remove_comments():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, ignore_comments=True)
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"], config=config)
    assert "# comment1" not in result
    assert "# comment2" not in result

def test_import_statement_wrap_length_overrides_line_length():
    config = Config(line_length=100, wrap_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from module", ["item1", "item2", "item3", "item4"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_explode_overrides_config():
    config = Config(line_length=100, multi_line_output=Modes.GRID)
    result = import_statement("from module", ["item1", "item2"], config=config, explode=True)
    assert result == "from module import (\n    item1,\n    item2,\n)"


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_65_false():
    config = Config()
    config.comment_prefix = "# "
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.wrap_length = None
    config.line_length = 80
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.indent = "    "
    line_separator = "\n"
    content = "from module import something"
    splitter = "import "
    comment = "noqa"
    noqa_comment = "# noqa"
    cont_line = "    another_thing"
    _comma = ","
    _separator = "\n"
    output = f"{content}{splitter}({noqa_comment}{line_separator}{cont_line}{_comma}{_separator})"
    lines = output.split(line_separator)
    result = config.comment_prefix in lines[-1] and lines[-1].endswith(")")
    assert result == False


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #20
#--------------------------

```python
def test_balanced_wrapping_predicate_evaluates_true():
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="# ", include_trailing_comma=False, ignore_comments=False, multi_line_output=Modes.GRID)
    import_start = "from module import"
    from_imports = ["very_long_import_name_that_will_cause_wrapping", "another_import", "third_import", "fourth_import", "fifth_import"]
    comments = []
    line_separator = "\n"
    multi_line_output = None
    explode = False
    statement = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    lines = statement.split(line_separator)
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if len(lines) > 1 else 0
    condition = len(lines[-1]) < minimum_length and len(lines) == line_count and config.wrap_length > 10
    assert condition == True


# LLM-generated content at query #21
#--------------------------

def test_include_trailing_comma_with_parentheses_and_no_comma_at_end():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    config.comment_prefix = "  # "
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 80
    config.wrap_length = None
    content = "from module import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "," in result


# LLM-generated content at query #22
#--------------------------

```python
def test_balanced_wrapping_predicate_evaluates_true():
    import_start = "from module import "
    from_imports = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="//", ignore_comments=False, include_trailing_comma=False, multi_line_output=Modes.GRID)
    statement = import_statement(import_start, from_imports, config=config, multi_line_output=Modes.GRID)
    lines = statement.split("\n")
    line_count = len(lines)
    minimum_length = min(len(line) for line in lines[:-1]) if line_count > 1 else 0
    predicate = len(lines[-1]) < minimum_length and len(lines) == line_count and config.wrap_length > 10
    assert predicate == True


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_71_is_false():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "some_very_long_import_statement_that_exceeds_line_length_by_a_lot # NOQA"
    result = line(content, "\n", config)
    assert not (len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content)


# LLM-generated content at query #24
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=100, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import function  # some comment", "\n", config)
    assert "from module import (" in result
    assert "function  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import function  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "function" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("from module import function", "\n", config)
    assert "from module import \\" in result
    assert "function" in result

def test_line_noqa_mode():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import function", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import function  # NOQA", "\n", config)
    assert result == "from module import function  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True)
    result = line("from module import function", "\n", config)
    assert "from module import (" in result
    assert "function," in result

def test_line_wrap_with_comment_and_trailing_comma():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import function  # comment", "\n", config)
    assert "from module import (" in result
    assert "function,  # comment" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, indent="    ", wrap_length=20, include_trailing_comma=False)
    result = line("from module import function", "\n", config)
    assert "from module import (" in result
    assert "function" in result

def test_line_wrap_with_empty_content():
    result = line("", "\n", Config(line_length=10, multi_line_output=Modes.GRID))
    assert result == ""


# LLM-generated content at query #25
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #26
#--------------------------

def test_predicate_at_line_43_true():
    config = Config()
    config.use_parentheses = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 10
    config.wrap_length = None
    config.indent = "    "
    config.comment_prefix = "  # "
    config.include_trailing_comma = True
    content = "verylongimportname"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert config.use_parentheses == True


# LLM-generated content at query #27
#--------------------------

def test_predicate_at_line_43_true():
    config = Config()
    config.use_parentheses = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 80
    config.wrap_length = None
    config.indent = "    "
    config.include_trailing_comma = True
    config.comment_prefix = "  # "
    content = "from very_long_module_name import very_long_submodule_name"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "as " not in content


# LLM-generated content at query #28
#--------------------------

def test_include_trailing_comma_with_parentheses_and_no_ending_comma():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    config.comment_prefix = "  # "
    content = "from module import something"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "," in result


# LLM-generated content at query #29
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=80, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False)
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something" in result
    assert "  # some comment" in result

def test_line_wrap_with_noqa_comment_and_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=False, comment_prefix="  # ")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (  # noqa" in result
    assert "something" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=None)
    result = line("from module import something", "\n", config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_wrap_mode_noqa():
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_mode_noqa_with_existing_noqa():
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_with_comment_and_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=None, include_trailing_comma=True, comment_prefix="  # ")
    result = line("from module import something  # comment", "\n", config)
    assert "from module import (" in result
    assert "something,  # comment" in result


# LLM-generated content at query #30
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import(" in result
    assert "    very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "    very_long_alias" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line("very_long_module_name.very_long_submodule.very_long_function", "\n", config)
    assert "very_long_module_name.(" in result
    assert "    very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import function  # some comment", "\n", config)
    assert "from module import(" in result
    assert "    function,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import function  # noqa", "\n", config)
    assert "from module import(  # noqa" in result
    assert "    function," in result

def test_line_noqa_mode_with_long_line():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    config = Config(line_length=20, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=Modes.GRID, wrap_length=20, use_parentheses=False, indent="    ", comment_prefix="  # ")
    result = line("from module import very_long_function_name", "\n", config)
    assert "from module import\\" in result
    assert "    very_long_function_name" in result

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import function  # comment", "\n", config)
    assert "from module import(" in result
    assert "    function,  # comment" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, wrap_length=20, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line("from module import function", "\n", config)
    assert "from module import(" in result
    assert "    function," in result


