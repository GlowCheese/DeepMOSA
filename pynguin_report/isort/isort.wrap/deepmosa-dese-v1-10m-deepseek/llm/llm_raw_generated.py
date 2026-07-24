####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=80, multi_line_output=3))
    assert result == "import os"

def test_line_wrap_with_comment():
    config = Config(line_length=20, multi_line_output=3, comment_prefix="  #", use_parentheses=True, include_trailing_comma=True)
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=10, multi_line_output=5, comment_prefix="  #")
    result = line("import verylongmodule", "\n", config)
    assert "NOQA" in result

def test_line_wrap_with_as_keyword():
    config = Config(line_length=20, multi_line_output=3, use_parentheses=True, include_trailing_comma=True)
    result = line("import something as something_else", "\n", config)
    assert "as" in result

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=20, multi_line_output=3, use_parentheses=True, include_trailing_comma=True)
    result = line("from package.subpackage import module", "\n", config)
    assert "from package.subpackage import (" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=20, multi_line_output=3, use_parentheses=True, include_trailing_comma=True)
    result = line("cimport numpy as np", "\n", config)
    assert "cimport" in result

def test_line_wrap_vertical_hanging_indent():
    config = Config(line_length=20, multi_line_output=4, use_parentheses=True, include_trailing_comma=True)
    result = line("from module import something, another_thing", "\n", config)
    assert "from module import (" in result

def test_line_wrap_vertical_grid_grouped():
    config = Config(line_length=20, multi_line_output=5, use_parentheses=True, include_trailing_comma=True)
    result = line("from module import something, another_thing", "\n", config)
    assert "from module import (" in result

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=20, multi_line_output=3, use_parentheses=True, comment_prefix="  #")
    result = line("from module import something  # noqa", "\n", config)
    assert "noqa" in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=20, multi_line_output=3, use_parentheses=False)
    result = line("from module import something", "\n", config)
    assert "\\" in result


# LLM-generated content at query #2
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #3
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
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    content = "very_long_module_name.very_long_submodule.very_long_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    content = "from module import something  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something  # some comment" in result

def test_line_wrap_noqa_mode():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result == "from very_long_module_name import very_long_function_name  # NOQA"

def test_line_wrap_existing_noqa():
    content = "from very_long_module_name import very_long_function_name  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result == "from very_long_module_name import very_long_function_name  # NOQA"

def test_line_wrap_with_trailing_comma():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_with_noqa_comment_and_parentheses():
    content = "from module import something  # noqa"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=True, indent="    ", comment_prefix="  # ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (  # noqa" in result
    assert "something" in result

def test_line_wrap_without_parentheses():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, wrap_length=30, use_parentheses=False, indent="    ")
    result = line(content, line_separator, config)
    assert "from module import \\" in result
    assert "something" in result


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_30_true():
    config = Config()
    config.wrap_length = 50
    config.line_length = 100
    content = "a" * 95
    line_parts = ["part1", "part2", "part3"]
    result = (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts
    assert result == True


# LLM-generated content at query #5
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts
    assert result is False


# LLM-generated content at query #6
#--------------------------

def test_predicate_at_line_11_true():
    import re
    from isort import Config, Modes
    line_without_comment = "from module import something"
    splitter = "import "
    exp = r"\b" + re.escape(splitter) + r"\b"
    result = re.search(exp, line_without_comment) and not line_without_comment.strip().startswith(splitter)
    assert result == True


# LLM-generated content at query #7
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    config = Config(line_length=100, multi_line_output=Modes.NOQA)
    result = line("some short content", "\n", config)
    assert result == "some short content"


# LLM-generated content at query #8
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
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"], config=config)
    assert "comment1" in result and "comment2" in result

def test_import_statement_balanced_wrapping():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=30, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False, balanced_wrapping=True)
    result = import_statement("from module", ["very_long_import_name1", "very_long_import_name2", "very_long_import_name3"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_single_line_wrap():
    config = Config(multi_line_output=Modes.GRID, line_length=10, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.count("\n") >= 1

def test_import_statement_custom_multi_line_output():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=80, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3", "item4"], multi_line_output=Modes.GRID, config=config)
    assert "from module import item1, item2, item3, item4" in result

def test_import_statement_no_wrap_needed():
    config = Config(multi_line_output=Modes.GRID, line_length=80, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1"], config=config)
    assert result == "from module import item1"

def test_import_statement_with_trailing_comma():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    assert result.endswith(",\n)")

def test_import_statement_line_separator():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20, wrap_length=None, include_trailing_comma=True, indent="    ", comment_prefix="  #", ignore_comments=False)
    result = import_statement("from module", ["item1", "item2", "item3"], line_separator="\r\n", config=config)
    assert "\r\n" in result

def test_import_statement_remove_comments():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=80, wrap_length=None, include_trailing_comma=False, indent="    ", comment_prefix="  #", ignore_comments=True)
    result = import_statement("from module", ["item1", "item2"], comments=["comment"], config=config)
    assert "comment" not in result


# LLM-generated content at query #9
#--------------------------

def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_wrap_with_import_splitter():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=40, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_dot_splitter():
    content = "module.submodule.very_long_submodule_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "module.submodule.(" in result
    assert "very_long_submodule_name" in result

def test_line_wrap_with_as_splitter():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=30, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "import very_long_module_name as vlm" in result

def test_line_wrap_with_comment():
    content = "from module import something  # some comment"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=40, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something" in result
    assert "# some comment" in result

def test_line_wrap_noqa_mode():
    content = "import very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result == "import very_long_module_name_that_exceeds_line_length  # NOQA"

def test_line_wrap_with_trailing_comma():
    content = "from module import item1, item2, item3"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=40, use_parentheses=True, indent="    ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "item1," in result
    assert "item2," in result
    assert "item3" in result

def test_line_wrap_with_noqa_comment():
    content = "from module import something  # noqa"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, wrap_length=40, use_parentheses=True, indent="    ", comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert "from module import (  # noqa" in result
    assert "something" in result

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import item1, item2, item3"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_GRID_GROUPED, wrap_length=40, use_parentheses=True, indent="    ")
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "item1" in result
    assert "item2" in result
    assert "item3" in result

def test_line_wrap_without_parentheses():
    content = "from module import very_long_item_name"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.GRID, wrap_length=40, use_parentheses=False, indent="    ")
    result = line(content, line_separator, config)
    assert "from module import \\" in result
    assert "very_long_item_name" in result


# LLM-generated content at query #10
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
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name," in result

def test_line_wrap_with_as_split():
    content = "import very_long_module_name as very_long_alias"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "import very_long_module_name as (" in result
    assert "very_long_alias" in result

def test_line_wrap_with_dot_split():
    content = "very_long_module_name.very_long_submodule.very_long_function"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule.very_long_function" in result

def test_line_wrap_with_comment():
    content = "from module import something  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something,  # some comment" in result

def test_line_wrap_with_noqa_comment():
    content = "from module import something  # noqa"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert "from module import (  # noqa" in result
    assert "something," in result

def test_line_wrap_without_parentheses():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", wrap_length=30, use_parentheses=False, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import \\" in result
    assert "something" in result

def test_line_noqa_mode():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result == "from very_long_module_name import very_long_function_name  # NOQA"

def test_line_noqa_mode_with_existing_noqa():
    content = "from very_long_module_name import very_long_function_name  # NOQA"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result == "from very_long_module_name import very_long_function_name  # NOQA"

def test_line_vertical_hanging_indent_mode():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something," in result

def test_line_vertical_grid_grouped_mode():
    content = "from module import something"
    line_separator = "\n"
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_GRID_GROUPED, indent="    ", wrap_length=30, use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "something," in result


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result is False


# LLM-generated content at query #12
#--------------------------

def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_wrap_with_comment():
    content = "import very_long_module_name_that_exceeds_line_length  # some comment"
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.GRID, comment_prefix="  # ", use_parentheses=True, include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "very_long_module_name_that_exceeds_line_length" in result
    assert "some comment" in result

def test_line_wrap_with_noqa_mode():
    content = "import very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, line_separator, config)
    assert result.endswith("NOQA")

def test_line_wrap_with_splitter_import():
    content = "from very_long_package_name import very_long_module_name"
    line_separator = "\n"
    config = Config(line_length=50, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, wrap_length=50, indent="    ")
    result = line(content, line_separator, config)
    assert "very_long_package_name" in result
    assert "very_long_module_name" in result

def test_line_wrap_with_as_splitter():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=False, wrap_length=40, indent="    ")
    result = line(content, line_separator, config)
    assert "very_long_module_name" in result
    assert "as vlm" in result

def test_line_wrap_with_dot_splitter():
    content = "very_long_module_name.very_long_submodule_name"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, include_trailing_comma=True, wrap_length=40, indent="    ")
    result = line(content, line_separator, config)
    assert "very_long_module_name" in result
    assert "very_long_submodule_name" in result

def test_line_wrap_with_noqa_in_comment():
    content = "import very_long_module_name  # noqa"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.GRID, use_parentheses=True, comment_prefix="  # ", include_trailing_comma=True)
    result = line(content, line_separator, config)
    assert "noqa" in result
    assert "very_long_module_name" in result

def test_line_wrap_without_parentheses():
    content = "import very_long_module_name"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.GRID, use_parentheses=False, wrap_length=40, indent="    ")
    result = line(content, line_separator, config)
    assert "\\" in result
    assert "very_long_module_name" in result

def test_line_wrap_with_comment_and_trailing_comma():
    content = "import very_long_module_name  # comment"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", wrap_length=40, indent="    ")
    result = line(content, line_separator, config)
    assert "," in result
    assert "comment" in result

def test_line_wrap_comment_inside_parentheses():
    content = "import very_long_module_name  # comment with ) inside"
    line_separator = "\n"
    config = Config(line_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False, comment_prefix="  # ", wrap_length=40, indent="    ")
    result = line(content, line_separator, config)
    assert "comment with ) inside" in result
    assert result.count(")") == 1


# LLM-generated content at query #13
#--------------------------

```python
def test_formatter_from_string_returns_grid_for_unknown_name():
    result = formatter_from_string("unknown_name")
    assert result == grid


# LLM-generated content at query #14
#--------------------------

def test_predicate_at_line_11_evaluates_to_true():
    import re
    from isort import Config
    from isort._line import _wrap_line
    from isort._line import line
    from isort._line import Modes
    config = Config(line_length=80, wrap_length=80, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    content = "from module import submodule as alias"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result is not None
    config2 = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    content2 = "import verylongmodulename"
    result2 = line(content2, line_separator, config2)
    assert result2 is not None
    config3 = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=False, comment_prefix="  # ")
    content3 = "from pkg import mod"
    result3 = line(content3, line_separator, config3)
    assert result3 is not None
    config4 = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    content4 = "cimport numpy as np"
    result4 = line(content4, line_separator, config4)
    assert result4 is not None
    config5 = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ")
    content5 = "import something  # noqa"
    result5 = line(content5, line_separator, config5)
    assert result5 is not None


# LLM-generated content at query #15
#--------------------------

```python
def test_formatter_from_string_returns_grid_for_unknown_name():
    result = formatter_from_string("unknown_name")
    assert result == grid


# LLM-generated content at query #16
#--------------------------

def test_import_statement_explode_mode():
    result = import_statement("from module", ["item1", "item2"], explode=True)
    expected = "from module import (\n    item1,\n    item2,\n)"
    assert result == expected

def test_import_statement_default_config():
    config = Config()
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert result == "from module import item1, item2"

def test_import_statement_with_comments():
    config = Config()
    result = import_statement("from module", ["item1"], comments=["comment"], config=config)
    assert result == "from module import item1  # comment"

def test_import_statement_multi_line_output():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = import_statement("from module", ["item1", "item2", "item3"], config=config)
    expected = "from module import (\n    item1,\n    item2,\n    item3,\n)"
    assert result == expected

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from module", ["item1", "item2", "item3", "item4", "item5"], config=config)
    lines = result.split("\n")
    assert len(lines) > 1

def test_import_statement_include_trailing_comma():
    config = Config(include_trailing_comma=True, multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20)
    result = import_statement("from module", ["item1", "item2"], config=config)
    expected = "from module import (\n    item1,\n    item2,\n)"
    assert result == expected

def test_import_statement_custom_line_separator():
    config = Config()
    result = import_statement("from module", ["item1", "item2"], line_separator="\r\n", config=config)
    assert result == "from module import item1, item2"

def test_import_statement_single_line_wrap():
    config = Config(line_length=10)
    result = import_statement("from module", ["item1", "item2"], config=config)
    assert result == "from module import item1, item2"

def test_import_statement_remove_comments():
    config = Config(ignore_comments=True)
    result = import_statement("from module", ["item1"], comments=["comment"], config=config)
    assert result == "from module import item1"

def test_import_statement_empty_from_imports():
    config = Config()
    result = import_statement("from module", [], config=config)
    assert result == "from module import "


# LLM-generated content at query #17
#--------------------------

def test_line_no_wrap_needed():
    result = line("import os", "\n", Config(line_length=80, multi_line_output=Modes.GRID))
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=20)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=30)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=30)
    result = line("very_long_module_name.very_long_submodule", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=30, comment_prefix="  #")
    result = line("from module import something  # some comment", "\n", config)
    assert "from module import (" in result
    assert "something" in result
    assert "# some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=30, comment_prefix="  #")
    result = line("from module import something  # noqa", "\n", config)
    assert "from module import (" in result
    assert "something" in result
    assert "# noqa" in result

def test_line_wrap_noqa_mode():
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_noqa_mode_with_existing_noqa():
    config = Config(line_length=30, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("from module import something  # NOQA", "\n", config)
    assert result == "from module import something  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, indent="    ", wrap_length=30, include_trailing_comma=True)
    result = line("from module import something", "\n", config)
    assert "from module import (" in result
    assert "something," in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=30, multi_line_output=Modes.GRID, use_parentheses=False, indent="    ", wrap_length=30)
    result = line("from module import something", "\n", config)
    assert "from module import \\" in result
    assert "something" in result


# LLM-generated content at query #18
#--------------------------

def test_predicate_at_line_71_true():
    config = Config()
    config.line_length = 10
    config.multi_line_output = Modes.NOQA
    result = line("very_long_line_content", "\n", config)
    assert result == "very_long_line_content# NOQA"


# LLM-generated content at query #19
#--------------------------

def test_predicate_at_line_42_evaluates_to_true():
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


# LLM-generated content at query #20
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=25, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("very.long.package.path.to.module", "\n", config)
    assert "very.long.package.path.to.(" in result
    assert "module" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("import long_module  # some comment", "\n", config)
    assert "import long_module  # some comment" == result

def test_line_wrap_noqa_mode():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import verylongmodule", "\n", config)
    assert result == "import verylongmodule#  NOQA"

def test_line_wrap_noqa_present():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import verylongmodule  # NOQA", "\n", config)
    assert result == "import verylongmodule  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", use_parentheses=True, include_trailing_comma=True, comment_prefix="# ")
    result = line("from module import very_long_name", "\n", config)
    assert "from module import (" in result
    assert "very_long_name," in result

def test_line_wrap_with_comment_and_trailing_comma():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", use_parentheses=True, include_trailing_comma=True, comment_prefix="# ")
    result = line("from module import long_name  # comment", "\n", config)
    assert "from module import (" in result
    assert "long_name,  # comment" in result

def test_line_wrap_with_noqa_comment_and_parentheses():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", use_parentheses=True, include_trailing_comma=True, comment_prefix="# ")
    result = line("from module import long_name  # noqa", "\n", config)
    assert "from module import (# noqa" in result
    assert "long_name," in result


# LLM-generated content at query #21
#--------------------------

def test_predicate_at_line_42_evaluates_to_true():
    config = Config()
    config.use_parentheses = True
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 80
    config.wrap_length = None
    config.indent = "    "
    config.comment_prefix = "  # "
    config.include_trailing_comma = True
    content = "from very_long_module_name import very_long_submodule_name"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert config.use_parentheses


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.line_length = 80
    config.wrap_length = None
    content = "short_line"
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert result == False


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_15_true():
    config = Config()
    config.use_parentheses = False
    comment = "# noqa"
    result = comment and not (config.use_parentheses and "noqa" in comment)
    assert result == True


# LLM-generated content at query #24
#--------------------------

def test_predicate_at_line_11_true():
    import re
    from isort import Config
    from isort._line import line
    from isort._line import Modes
    config = Config()
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 10
    config.wrap_length = 10
    content = "from module import something"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result is not None
    assert result != content
    assert "import" in result


# LLM-generated content at query #25
#--------------------------

def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=80, wrap_length=None, multi_line_output=3, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_noqa_wrap_mode():
    content = "import very_long_module_name_that_exceeds_line_length_by_a_lot"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=None, multi_line_output=5, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import very_long_module_name_that_exceeds_line_length_by_a_lot # NOQA"

def test_line_noqa_present_no_extra():
    content = "import os  # NOQA"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os  # NOQA"

def test_line_wrap_with_import_splitter():
    content = "from very_long_package_name import very_long_module_name"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=50, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "import" in result
    assert line_separator in result

def test_line_wrap_with_dot_splitter():
    content = "very_long_package_name.very_long_module_name.very_long_attribute"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=50, multi_line_output=3, use_parentheses=True, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "." in result
    assert line_separator in result

def test_line_wrap_with_as_splitter():
    content = "import very_long_module_name as very_long_alias_name"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=50, multi_line_output=3, use_parentheses=True, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "as" in result
    assert line_separator in result

def test_line_wrap_with_comment():
    content = "from package import module  # some comment"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=30, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "# some comment" in result
    assert line_separator in result

def test_line_wrap_with_noqa_comment_and_parentheses():
    content = "from package import module  # noqa"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=30, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "# noqa" in result
    assert line_separator in result

def test_line_wrap_vertical_hanging_indent():
    content = "from very_long_package_name import very_long_module_name, another_module"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=50, multi_line_output=4, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert line_separator in result
    assert result.count(line_separator) >= 1

def test_line_wrap_vertical_grid_grouped():
    content = "from very_long_package_name import very_long_module_name, another_module"
    line_separator = "\n"
    config = Config(line_length=50, wrap_length=50, multi_line_output=5, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert line_separator in result
    assert result.count(line_separator) >= 1

def test_line_wrap_without_parentheses():
    content = "from package import module"
    line_separator = "\n"
    config = Config(line_length=20, wrap_length=20, multi_line_output=3, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "\\" + line_separator in result

def test_line_wrap_with_trailing_comma_and_comment():
    content = "from package import module  # comment"
    line_separator = "\n"
    config = Config(line_length=30, wrap_length=30, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert "# comment" in result
    assert result.rstrip().endswith(",") or "# comment" in result

def test_line_no_wrap_due_to_noqa_mode_but_no_noqa_comment():
    content = "import very_long_name"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert result.endswith(" # NOQA")

def test_line_starts_with_splitter_no_wrap():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=5, wrap_length=None, multi_line_output=3, use_parentheses=False, include_trailing_comma=False, comment_prefix=" # ", indent="    ")
    result = line(content, line_separator, config)
    assert result == "import os"


# LLM-generated content at query #26
#--------------------------

def test_predicate_at_line_4_evaluates_to_false():
    config = Config()
    config.line_length = 100
    config.multi_line_output = Modes.NOQA
    result = len("short_content") > config.line_length and config.multi_line_output != Modes.NOQA
    assert result == False


# LLM-generated content at query #27
#--------------------------

def test_balanced_wrapping_condition_true():
    config = Config(balanced_wrapping=True, wrap_length=50, line_length=50, indent="    ", comment_prefix="# ", include_trailing_comma=False, ignore_comments=False, multi_line_output=Modes.GRID)
    import_start = "from module import"
    from_imports = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
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


# LLM-generated content at query #28
#--------------------------

def test_predicate_at_line_71_true():
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.NOQA
    content = "a" * 90
    result = line(content, "\n", config)
    assert result == content + config.comment_prefix + " NOQA"


# LLM-generated content at query #29
#--------------------------

def test_predicate_at_line_17_true():
    config = Config()
    config.include_trailing_comma = True
    config.use_parentheses = True
    line_without_comment = "something"
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


# LLM-generated content at query #30
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "import" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("module.submodule.verylongclass.verylongmethod", "\n", config)
    assert "." in result
    assert "verylongmethod" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=25, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "as" in result
    assert "vlm" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name  # some comment", "\n", config)
    assert "# some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert "# NOQA" in result

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name  # NOQA", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

def test_line_wrap_with_parentheses_and_trailing_comma():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=True)
    result = line("from module import very_long_function_name", "\n", config)
    assert ")" in result
    assert "," in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=False, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert "\\" in result

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert "(" in result
    assert ")" in result

def test_line_wrap_with_cimport():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("cimport very_long_module_name", "\n", config)
    assert "cimport" in result

def test_line_wrap_with_wrap_length():
    config = Config(line_length=80, wrap_length=30, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", use_parentheses=True, include_trailing_comma=False)
    result = line("import very_long_module_name", "\n", config)
    assert "very_long_module_name" in result


# LLM-generated content at query #31
#--------------------------

def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_import_split():
    config = Config(line_length=20, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "from very_long_module_name import (" in result
    assert "very_long_function_name" in result

def test_line_wrap_with_as_split():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_split():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("very_long_module_name.very_long_submodule", "\n", config)
    assert "very_long_module_name.(" in result
    assert "very_long_submodule" in result

def test_line_wrap_with_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name  # some comment", "\n", config)
    assert "import very_long_module_name  # some comment" in result

def test_line_wrap_with_noqa_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name# NOQA"

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name  # NOQA", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", use_parentheses=True, include_trailing_comma=True, comment_prefix="# ")
    result = line("from module import very_long_function_name", "\n", config)
    assert "," in result

def test_line_wrap_without_parentheses():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=False, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name", "\n", config)
    assert "\\" in result

def test_line_wrap_with_comment_and_noqa_in_comment():
    config = Config(line_length=30, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", use_parentheses=True, include_trailing_comma=False, comment_prefix="# ")
    result = line("import very_long_module_name  # noqa", "\n", config)
    assert "# noqa" in result
    assert ")" in result


####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```
def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    assert line(content, line_separator, config) == "import os"

def test_line_wrap_with_comment():
    content = "import os  # comment"
    line_separator = "\n"
    config = Config(line_length=10, multi_line_output=Modes.GRID)
    assert line(content, line_separator, config) == "import os  # comment"

def test_line_wrap_with_splitter():
    content = "from module.submodule import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True)
    expected = "from module.submodule import (very_long_name_that_exceeds_line_length,)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_noqa():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=5, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "import os# NOQA"

def test_line_wrap_with_as_splitter():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.GRID, use_parentheses=True)
    expected = "import very_long_module_name as vlm"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_comment_and_noqa():
    content = "import os  # noqa"
    line_separator = "\n"
    config = Config(line_length=5, multi_line_output=Modes.GRID, use_parentheses=True)
    expected = "import os  # noqa"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_vertical_hanging_indent():
    content = "from module import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True)
    expected = "from module import (\n    very_long_name_that_exceeds_line_length)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_vertical_grid_grouped():
    content = "from module import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True)
    expected = "from module import (\n    very_long_name_that_exceeds_line_length\n)"
    assert line(content, line_separator, config) == expected


# LLM-generated content at query #2
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    content = "a" * 100
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert "NOQA" in result


# LLM-generated content at query #3
#--------------------------

def test_import_statement_basic():
    result = import_statement("from module", ["import1", "import2"])
    assert isinstance(result, str)

def test_import_statement_with_comments():
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert isinstance(result, str)
    assert "comment1" in result
    assert "comment2" in result

def test_import_statement_explode_mode():
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert isinstance(result, str)
    assert result.count("\n") >= 1

def test_import_statement_custom_line_separator():
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert isinstance(result, str)
    assert "\r\n" in result

def test_import_statement_single_line_output():
    result = import_statement("from module", ["import1"], line_length=1000)
    assert isinstance(result, str)
    assert result.count("\n") == 0

def test_import_statement_balanced_wrapping():
    config = Config()
    config.balanced_wrapping = True
    result = import_statement("from module", ["import1", "import2", "import3", "import4", "import5"], config=config)
    assert isinstance(result, str)
    assert result.count("\n") > 0

def test_import_statement_with_trailing_comma():
    config = Config()
    config.include_trailing_comma = True
    result = import_statement("from module", ["import1", "import2"], config=config)
    assert isinstance(result, str)
    assert result.strip().endswith(",")


# LLM-generated content at query #4
#--------------------------

def test_import_statement_basic():
    result = import_statement("from module", ["import1", "import2"])
    assert isinstance(result, str)

def test_import_statement_with_comments():
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert isinstance(result, str)
    assert "comment1" in result
    assert "comment2" in result

def test_import_statement_explode_mode():
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert isinstance(result, str)
    assert result.count("\n") >= 1

def test_import_statement_custom_line_separator():
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert isinstance(result, str)
    assert "\r\n" in result

def test_import_statement_balanced_wrapping():
    config = Config(balanced_wrapping=True, wrap_length=40)
    result = import_statement("from module", ["import1", "import2", "import3", "import4"], config=config)
    assert isinstance(result, str)
    assert result.count("\n") > 0

def test_import_statement_single_line_wrap():
    config = Config(wrap_length=1000)
    result = import_statement("from module", ["import1", "import2"], config=config)
    assert isinstance(result, str)
    assert result.count("\n") == 0

def test_import_statement_empty_imports():
    result = import_statement("from module", [])
    assert isinstance(result, str)
    assert "from module" in result

def test_import_statement_with_trailing_comma():
    config = Config(include_trailing_comma=True)
    result = import_statement("from module", ["import1", "import2"], config=config)
    assert isinstance(result, str)
    assert "," in result.split("\n")[-1]


# LLM-generated content at query #5
#--------------------------

```python
def test_regex_search_and_not_startswith_splitter():
    content = "import os"
    line_without_comment = "import os"
    splitter = "import "
    exp = r"\b" + re.escape(splitter) + r"\b"
    assert re.search(exp, line_without_comment) and not line_without_comment.strip().startswith(splitter)


# LLM-generated content at query #6
#--------------------------

```
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=0, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    assert line("import os", "\n", config) == "import os"

def test_line_with_comment_no_wrap():
    config = Config(line_length=80, wrap_length=None, multi_line_output=0, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    assert line("import os  # comment", "\n", config) == "import os  # comment"

def test_line_wrap_with_import_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("from module import thing", "\n", config) == "from module import(\n    thing,"

def test_line_wrap_with_as_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("import thing as other", "\n", config) == "import thing as other"

def test_line_wrap_with_dot_split():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("module.submodule.thing", "\n", config) == "module.submodule.thing"

def test_line_wrap_with_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    assert line("verylongimportname", "\n", config) == "verylongimportname  # NOQA"

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, use_parentheses=False, include_trailing_comma=False, comment_prefix="  # ", indent="    ")
    assert line("verylongimportname  # NOQA", "\n", config) == "verylongimportname  # NOQA"

def test_line_wrap_with_comment_and_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("from module import thing  # noqa", "\n", config) == "from module import(\n    thing  # noqa,"

def test_line_wrap_vertical_grid():
    config = Config(line_length=10, wrap_length=None, multi_line_output=4, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("from module import thing", "\n", config) == "from module import(\n    thing\n,"

def test_line_wrap_vertical_hanging():
    config = Config(line_length=10, wrap_length=None, multi_line_output=2, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("from module import thing", "\n", config) == "from module import(\n    thing,"

def test_line_wrap_with_trailing_comma_no_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, use_parentheses=True, include_trailing_comma=True, comment_prefix="  # ", indent="    ")
    assert line("from module import thing", "\n", config) == "from module import(\n    thing,"


# LLM-generated content at query #7
#--------------------------

```python
def test_predicate_at_line_11_evaluates_to_true():
    content = "import module"
    line_without_comment = "import module"
    splitter = "import "
    exp = r"\b" + "import " + r"\b"
    assert re.search(exp, line_without_comment) and not line_without_comment.strip().startswith(splitter)


# LLM-generated content at query #8
#--------------------------

```
def test_predicate_at_line_11_evaluates_to_true():
    content = "from module import something"
    line_without_comment = "from module import something"
    splitter = "import "
    assert re.search(r"\b" + re.escape(splitter) + r"\b", line_without_comment) and not line_without_comment.strip().startswith(splitter)


# LLM-generated content at query #9
#--------------------------

```python
def test_wrap_length_used_when_set():
    config = Config(wrap_length=100, line_length=120)
    line_length = config.wrap_length or config.line_length
    assert line_length == 100

def test_line_length_used_when_wrap_length_not_set():
    config = Config(wrap_length=None, line_length=120)
    line_length = config.wrap_length or config.line_length
    assert line_length == 120


# LLM-generated content at query #10
#--------------------------

```python
def test_balanced_wrapping_with_multiple_lines():
    import_start = "from module import"
    from_imports = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    comments = []
    line_separator = "\n"
    config = Config(
        multi_line_output=Modes.GRID,
        wrap_length=20,
        line_length=20,
        include_trailing_comma=True,
        balanced_wrapping=True,
        indent="    ",
        comment_prefix="#",
        ignore_comments=False,
    )
    statement = import_statement(
        import_start=import_start,
        from_imports=from_imports,
        comments=comments,
        line_separator=line_separator,
        config=config,
        multi_line_output=Modes.GRID,
    )
    lines = statement.split(line_separator)
    assert len(lines) > 1
    minimum_length = min(len(line) for line in lines[:-1])
    assert len(lines[-1]) < minimum_length
    assert len(lines) == len(lines)
    line_length = config.line_length
    assert line_length > 10


# LLM-generated content at query #11
#--------------------------

def test_import_statement_basic():
    result = import_statement("from module", ["import1", "import2"])
    assert isinstance(result, str)

def test_import_statement_with_comments():
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert isinstance(result, str)
    assert "comment1" in result
    assert "comment2" in result

def test_import_statement_explode_mode():
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert isinstance(result, str)
    assert result.count("\n") >= 1

def test_import_statement_custom_line_separator():
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert isinstance(result, str)
    assert "\r\n" in result

def test_import_statement_single_import():
    result = import_statement("from module", ["import1"])
    assert isinstance(result, str)
    assert result.count("\n") == 0

def test_import_statement_multiple_imports():
    result = import_statement("from module", ["import1", "import2", "import3", "import4"])
    assert isinstance(result, str)
    assert result.count("\n") >= 1

def test_import_statement_with_balanced_wrapping():
    class Config:
        balanced_wrapping = True
        wrap_length = 50
        line_length = 50
        include_trailing_comma = False
        indent = "    "
        ignore_comments = False
        comment_prefix = "#"
        multi_line_output = None
    
    result = import_statement("from very.long.module.path", ["import1", "import2", "import3", "import4"], config=Config())
    assert isinstance(result, str)
    assert result.count("\n") >= 1


# LLM-generated content at query #12
#--------------------------

```
def test_line_no_wrap_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    assert line(content, line_separator, config) == "import os"

def test_line_wrap_with_comment():
    content = "from very_long_module_name import very_long_function_name  # some comment"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    expected = "from very_long_module_name import (\n    very_long_function_name  # some comment,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_noqa():
    content = "from very_long_module_name import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    assert line(content, line_separator, config) == "from very_long_module_name import very_long_function_name# NOQA"

def test_line_wrap_with_as_keyword():
    content = "from very_long_module_name import very_long_function_name as vlf"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    expected = "from very_long_module_name import (\n    very_long_function_name as vlf,\n)"
    assert line(content, line_separator, config) == expected

def test_line_wrap_with_dot_separator():
    content = "very_long_module_name.very_long_submodule_name.very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=40, wrap_length=None, multi_line_output=Modes.GRID, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    expected = "very_long_module_name.very_long_submodule_name.\\\n    very_long_function_name"
    assert line(content, line_separator, config) == expected


# LLM-generated content at query #13
#--------------------------

Here's the unit test for the predicate at line 17:


# LLM-generated content at query #14
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    config = Config(line_length=80, wrap_length=80, multi_line_output=Modes.NOQA, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    content = "a" * 81
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #15
#--------------------------

```python
def test_formatter_from_string_returns_default_grid_when_name_not_found():
    formatter = formatter_from_string("NON_EXISTENT_NAME")
    assert formatter == grid


# LLM-generated content at query #16
#--------------------------

```python
def test_line_no_wrapping_needed():
    config = Config(line_length=80, multi_line_output=Modes.NOQA)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed_with_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import os", "\n", config)
    assert result == "import os# NOQA"

def test_line_wrapping_needed_with_splitter():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import os.path", "\n", config)
    assert result == "import os.\\\n    path"

def test_line_wrapping_needed_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import os.path  # comment", "\n", config)
    assert result == "import os.(\n    path# comment)"


# LLM-generated content at query #17
#--------------------------

```python
def test_predicate_evaluates_to_false_when_content_length_is_within_limit():
    class Config:
        def __init__(self, line_length, multi_line_output):
            self.line_length = line_length
            self.multi_line_output = multi_line_output

    config = Config(line_length=10, multi_line_output="NOQA")
    content = "short"
    assert not (len(content) > config.line_length and config.multi_line_output != "NOQA")

def test_predicate_evaluates_to_false_when_wrap_mode_is_noqa():
    class Config:
        def __init__(self, line_length, multi_line_output):
            self.line_length = line_length
            self.multi_line_output = multi_line_output

    config = Config(line_length=10, multi_line_output="NOQA")
    content = "longer_than_line_length"
    assert not (len(content) > config.line_length and config.multi_line_output != "NOQA")


# LLM-generated content at query #18
#--------------------------

```python
def test_import_statement_with_explode():
    import_start = "from module"
    from_imports = ["import1", "import2", "import3"]
    comments = ["comment1", "comment2"]
    line_separator = "\n"
    config = DEFAULT_CONFIG
    multi_line_output = None
    explode = True
    result = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    assert isinstance(result, str)
    assert line_separator in result

def test_import_statement_without_explode():
    import_start = "from module"
    from_imports = ["import1", "import2", "import3"]
    comments = ["comment1", "comment2"]
    line_separator = "\n"
    config = DEFAULT_CONFIG
    multi_line_output = None
    explode = False
    result = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    assert isinstance(result, str)
    assert line_separator in result

def test_import_statement_with_balanced_wrapping():
    import_start = "from module"
    from_imports = ["import1", "import2", "import3", "import4", "import5"]
    comments = ["comment1", "comment2"]
    line_separator = "\n"
    config = DEFAULT_CONFIG
    config.balanced_wrapping = True
    multi_line_output = None
    explode = False
    result = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    assert isinstance(result, str)
    assert line_separator in result

def test_import_statement_with_single_line():
    import_start = "from module"
    from_imports = ["import1"]
    comments = ["comment1"]
    line_separator = "\n"
    config = DEFAULT_CONFIG
    multi_line_output = None
    explode = False
    result = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    assert isinstance(result, str)
    assert line_separator not in result


# LLM-generated content at query #19
#--------------------------

```python
def test_formatter_from_string_returns_grid_for_unknown_name():
    result = formatter_from_string("UNKNOWN_NAME")
    assert result == grid

def test_formatter_from_string_returns_correct_formatter_for_known_name():
    result = formatter_from_string("VERTICAL_HANGING_INDENT")
    assert result == vertical_hanging_indent


# LLM-generated content at query #20
#--------------------------

```python
def test_wrap_length_used_when_present():
    config = Config(wrap_length=100, line_length=80)
    line_length = config.wrap_length or config.line_length
    assert line_length == 100

def test_line_length_used_when_wrap_length_not_present():
    config = Config(wrap_length=None, line_length=80)
    line_length = config.wrap_length or config.line_length
    assert line_length == 80


# LLM-generated content at query #21
#--------------------------

```python
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix="# ", indent="    ")
    content = "import os"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os"

def test_line_wrap_needed():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    content = "import os, sys, math"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os,\n    sys,\n    math"

def test_line_wrap_needed_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    content = "import os, sys, math  # noqa"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os,\n    sys,\n    math # noqa"

def test_line_wrap_needed_noqa_mode():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix="# ", indent="    ")
    content = "import os, sys, math"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os, sys, math # NOQA"

def test_line_wrap_needed_noqa_mode_with_existing_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix="# ", indent="    ")
    content = "import os, sys, math  # NOQA"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os, sys, math  # NOQA"

def test_line_wrap_needed_with_as_keyword():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    content = "import os as operating_system"
    line_separator = "\n"
    assert line(content, line_separator, config) == "import os as operating_system"

def test_line_wrap_needed_with_dot_keyword():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ")
    content = "from os.path import join"
    line_separator = "\n"
    assert line(content, line_separator, config) == "from os.path\n    import join"


# LLM-generated content at query #22
#--------------------------

```python
def test_line_length_set_to_wrap_length_when_wrap_length_is_set():
    config = Mock(wrap_length=80, line_length=100, include_trailing_comma=False, indent="    ", ignore_comments=False, balanced_wrapping=False, comment_prefix="#")
    line_length = import_statement("from module", ["import1", "import2"], config=config).split("\n")[0].strip().count(" ")
    assert line_length == 80


# LLM-generated content at query #23
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    content = "a" * 100
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    assert len(content) > config.line_length and config.multi_line_output == Modes.NOQA and "# NOQA" not in content


# LLM-generated content at query #24
#--------------------------

```python
def test_import_statement_explode_mode():
    import_start = "from module import"
    from_imports = ["func1", "func2", "func3"]
    result = import_statement(import_start, from_imports, explode=True)
    assert result == "from module import (\n    func1,\n    func2,\n    func3,\n)"

def test_import_statement_default_mode():
    import_start = "from module import"
    from_imports = ["func1", "func2", "func3"]
    result = import_statement(import_start, from_imports)
    assert result == "from module import func1, func2, func3"

def test_import_statement_with_comments():
    import_start = "from module import"
    from_imports = ["func1", "func2", "func3"]
    comments = ["comment1", "comment2"]
    result = import_statement(import_start, from_imports, comments=comments)
    assert result == "from module import func1, func2, func3  # comment1  # comment2"

def test_import_statement_balanced_wrapping():
    import_start = "from module import"
    from_imports = ["func1", "func2", "func3", "func4", "func5"]
    result = import_statement(import_start, from_imports, config=Config(wrap_length=20, balanced_wrapping=True))
    assert result == "from module import func1, func2,\n    func3, func4,\n    func5"


# LLM-generated content at query #25
#--------------------------

```python
def test_formatter_from_string_returns_grid_when_name_not_found():
    result = formatter_from_string("INVALID_NAME")
    assert result == grid


# LLM-generated content at query #26
#--------------------------

```python
def test_formatter_from_string_with_invalid_name():
    result = formatter_from_string("INVALID_FORMATTER_NAME")
    assert result == grid


# LLM-generated content at query #27
#--------------------------

```python
def test_formatter_from_string_with_invalid_name():
    invalid_name = "INVALID_FORMATTER"
    assert formatter_from_string(invalid_name) == grid


# LLM-generated content at query #28
#--------------------------

def test_import_statement_basic():
    result = import_statement("from module import", ["func1", "func2"])
    assert isinstance(result, str)

def test_import_statement_explode():
    result = import_statement("from module import", ["func1", "func2"], explode=True)
    assert "\n" in result

def test_import_statement_with_comments():
    result = import_statement("from module import", ["func1", "func2"], comments=["comment1", "comment2"])
    assert isinstance(result, str)

def test_import_statement_custom_line_separator():
    result = import_statement("from module import", ["func1", "func2"], line_separator="\r\n")
    assert "\r\n" in result

def test_import_statement_single_import():
    result = import_statement("from module import", ["func1"])
    assert isinstance(result, str)

def test_import_statement_balanced_wrapping():
    config = Config()
    config.balanced_wrapping = True
    result = import_statement("from module import", ["func1", "func2", "func3", "func4", "func5"], config=config)
    assert isinstance(result, str)


# LLM-generated content at query #29
#--------------------------

```
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=0, indent="    ", comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line("import os", "\n", config) == "import os"

def test_line_with_comment_no_wrap():
    config = Config(line_length=80, wrap_length=None, multi_line_output=0, indent="    ", comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line("import os  # comment", "\n", config) == "import os  # comment"

def test_line_wrap_with_import():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    assert line("from very.long.package.path import module", "\n", config) == "from very.long.package.path import(\n    module,"

def test_line_wrap_with_cimport():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    assert line("from very.long.package.path cimport module", "\n", config) == "from very.long.package.path cimport(\n    module,"

def test_line_wrap_with_as():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    assert line("import verylongmodule as vlm", "\n", config) == "import verylongmodule as vlm"

def test_line_wrap_with_dot():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    assert line("very.long.package.path.module", "\n", config) == "very.long.package.path.module"

def test_line_wrap_with_noqa_mode():
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, indent="    ", comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line("verylongmodule", "\n", config) == "verylongmodule# NOQA"

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=5, indent="    ", comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line("verylongmodule  # NOQA", "\n", config) == "verylongmodule  # NOQA"

def test_line_wrap_with_comment_and_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    assert line("from very.long.path import module  # noqa", "\n", config) == "from very.long.path import(# noqa\n    module)"

def test_line_wrap_with_comment_and_no_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=False)
    assert line("from very.long.path import module  # comment", "\n", config) == "from very.long.path import\\\n    module  # comment"


# LLM-generated content at query #30
#--------------------------

```python
def test_import_statement_basic():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2"],
        config=Config(line_length=80, wrap_length=80)
    )
    assert result == "from module import func1, func2"

def test_import_statement_with_comments():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2"],
        comments=["comment1", "comment2"],
        config=Config(line_length=80, wrap_length=80)
    )
    assert "# comment1" in result
    assert "# comment2" in result

def test_import_statement_multi_line():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2", "func3", "func4", "func5"],
        config=Config(line_length=40, wrap_length=40)
    )
    assert "\n" in result
    assert "func1," in result
    assert "func5" in result

def test_import_statement_explode_mode():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2"],
        explode=True,
        config=Config(line_length=80, wrap_length=80)
    )
    assert "\n" in result
    assert "func1," in result
    assert "func2" in result

def test_import_statement_trailing_comma():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2"],
        config=Config(line_length=80, wrap_length=80, include_trailing_comma=True)
    )
    assert result.endswith("func2,")

def test_import_statement_balanced_wrapping():
    result = import_statement(
        import_start="from module import",
        from_imports=["func1", "func2", "func3", "func4", "func5", "func6", "func7"],
        config=Config(line_length=40, wrap_length=40, balanced_wrapping=True)
    )
    lines = result.split("\n")
    assert len(lines) > 1
    assert all(len(line) <= 40 for line in lines)


# LLM-generated content at query #31
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    config = Config(line_length=80, wrap_length=80, multi_line_output=Modes.NOQA, include_trailing_comma=True, use_parentheses=True, comment_prefix=" #", indent="    ")
    content = "import very_long_module_name_that_exceeds_the_line_length_by_a_lot"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert len(content) + 2 <= config.wrap_length or config.line_length


# LLM-generated content at query #32
#--------------------------

```python
def test_re_search_matches_and_not_starts_with_splitter():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=100, wrap_length=80, use_parentheses=True, include_trailing_comma=True)
    content = "from module import something"
    line_without_comment = content
    splitter = "import "
    exp = r"\b" + re.escape(splitter) + r"\b"
    assert re.search(exp, line_without_comment) and not line_without_comment.strip().startswith(splitter)


# LLM-generated content at query #33
#--------------------------

```python
def test_line_no_wrap_needed():
    content = "short line"
    line_separator = "\n"
    config = Config(line_length=80, wrap_length=None, comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line(content, line_separator, config) == "short line"

def test_line_wrap_with_comment():
    content = "long line with comment # comment"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, comment_prefix="# ", include_trailing_comma=False, use_parentheses=False)
    assert line(content, line_separator, config) == "long line with comment # comment"

def test_line_wrap_with_noqa():
    content = "very long line that needs wrapping # NOQA"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, comment_prefix="# ", include_trailing_comma=False, use_parentheses=False, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "very long line that needs wrapping # NOQA"

def test_line_wrap_with_parentheses():
    content = "from module import very_long_name"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "from module import (" in result
    assert "very_long_name," in result

def test_line_wrap_with_as_keyword():
    content = "import very_long_module_name as vlm"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "import very_long_module_name as (" in result
    assert "vlm" in result

def test_line_wrap_with_dot_operator():
    content = "module.submodule.very_long_name"
    line_separator = "\n"
    config = Config(line_length=10, wrap_length=None, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line(content, line_separator, config)
    assert "module.submodule.(" in result
    assert "very_long_name" in result


# LLM-generated content at query #34
#--------------------------

```python
def test_predicate_at_line_42_evaluates_to_true():
    config = Config(line_length=80, wrap_length=80, use_parentheses=True, include_trailing_comma=True, comment_prefix="#", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "from module import very_long_name_that_exceeds_the_line_length"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert config.use_parentheses


# LLM-generated content at query #35
#--------------------------

```
def test_line_no_wrapping_needed():
    config = Config(line_length=80, multi_line_output=Modes.NOQA)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as (\n    path # ,\n)"

def test_line_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" # ", indent="    ")
    result = line("import os.path as path # comment", "\n", config)
    assert result == "import os.path as (\n    path # comment,\n)"

def test_line_with_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix=" # ")
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as path # NOQA"

def test_line_with_noqa_and_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix=" # ")
    result = line("import os.path as path # comment", "\n", config)
    assert result == "import os.path as path # comment NOQA"


# LLM-generated content at query #36
#--------------------------

```python
def test_predicate_at_line_15_evaluates_to_true():
    class Config:
        use_parentheses = False
    config = Config()
    comment = "some comment"
    assert comment and not (config.use_parentheses and "noqa" in comment)


# LLM-generated content at query #37
#--------------------------

```python
def test_predicate_at_line_71_evaluates_to_true():
    content = "a" * 100  # content longer than line_length
    config = Config(line_length=50, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert "# NOQA" in result


# LLM-generated content at query #38
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    config = Config(line_length=50, wrap_length=40, indent="    ", use_parentheses=True, include_trailing_comma=True, comment_prefix="# ")
    content = "a" * 100
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert "a" * 100 != result


# LLM-generated content at query #39
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short_content"
    config = Config(line_length=100, wrap_length=50, multi_line_output=Modes.NOQA, use_parentheses=True, include_trailing_comma=True, comment_prefix="#", indent="    ")
    assert not (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #40
#--------------------------

```python
def test_predicate_evaluates_to_false():
    content = "short_line"
    config = Config(line_length=100, wrap_length=None)
    assert not (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #41
#--------------------------

```python
def test_line_no_wrapping_needed():
    config = Config(line_length=80, wrap_length=80, multi_line_output=Modes.NOQA, comment_prefix="# ", include_trailing_comma=False, use_parentheses=False, indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed_with_noqa():
    config = Config(line_length=10, wrap_length=10, multi_line_output=Modes.NOQA, comment_prefix="# ", include_trailing_comma=False, use_parentheses=False, indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os# NOQA"

def test_line_wrapping_needed_with_parentheses():
    config = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os(\n    )"

def test_line_wrapping_needed_with_comment():
    config = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    result = line("import os  # noqa", "\n", config)
    assert result == "import os(# noqa\n    )"

def test_line_wrapping_needed_with_splitter():
    config = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    result = line("import os as operating_system", "\n", config)
    assert result == "import os as operating_system"


# LLM-generated content at query #42
#--------------------------

```python
def test_trailing_comma_with_use_parentheses_and_include_trailing_comma():
    config = Config(use_parentheses=True, include_trailing_comma=True)
    content = "import some_module"
    comment = "# some comment"
    line_without_comment = content
    assert _comma_maybe == ","


# LLM-generated content at query #43
#--------------------------

```python
def test_line_without_wrapping():
    config = Config(line_length=100, multi_line_output=Modes.NOQA, comment_prefix=" #", use_parentheses=False)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_with_wrapping():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix=" #", use_parentheses=True, wrap_length=10, indent="    ", include_trailing_comma=True)
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as (\n    path,\n)"

def test_line_with_noqa():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix=" #", use_parentheses=False, wrap_length=10, indent="    ", include_trailing_comma=True)
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as path # NOQA"

def test_line_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix=" #", use_parentheses=True, wrap_length=10, indent="    ", include_trailing_comma=True)
    result = line("import os.path as path # noqa", "\n", config)
    assert result == "import os.path as (\n    path # noqa,\n)"

def test_line_with_long_content():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix=" #", use_parentheses=True, wrap_length=10, indent="    ", include_trailing_comma=True)
    result = line("import os.path as path, sys, math", "\n", config)
    assert result == "import os.path as (\n    path,\n    sys,\n    math,\n)"


# LLM-generated content at query #44
#--------------------------

```python
def test_line_no_wrap_needed():
    config = Config(line_length=100, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("import os.path as path  # noqa", "\n", config)
    assert result == "import os.path as path  # noqa"

def test_line_wrap_with_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from very.long.package.name import very_long_module_name", "\n", config)
    assert result == "from very.long.package.name import(\n    very_long_module_name,\n)"

def test_line_wrap_with_noqa_mode():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from very.long.package.name import very_long_module_name", "\n", config)
    assert result == "from very.long.package.name import very_long_module_name # NOQA"

def test_line_wrap_with_splitter():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as(\n    path\n)"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("import os.path as path", "\n", config)
    assert result == "import os.path as(\n    path,\n)"


# LLM-generated content at query #45
#--------------------------

```
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("import os", "\n", config) == "import os"

def test_line_wrap_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("import os # comment", "\n", config) == "import os # comment"

def test_line_wrap_with_splitter():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("from module import very_long_name", "\n", config) == "from module import(\n    very_long_name,)"

def test_line_wrap_with_as_splitter():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("import very_long_name as vln", "\n", config) == "import very_long_name as vln"

def test_line_wrap_with_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("import very_long_name", "\n", config) == "import very_long_name# NOQA"

def test_line_wrap_with_existing_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("import very_long_name # NOQA", "\n", config) == "import very_long_name # NOQA"

def test_line_wrap_with_dot_splitter():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("module.very_long_name", "\n", config) == "module.very_long_name"

def test_line_wrap_with_vertical_hanging_indent():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("from module import name1, name2, name3", "\n", config) == "from module import(\n    name1,\n    name2,\n    name3,)"

def test_line_wrap_with_vertical_grid_grouped():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_GRID_GROUPED, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("from module import name1, name2, name3", "\n", config) == "from module import(\n    name1,\n    name2,\n    name3,)"

def test_line_wrap_with_trailing_comma_and_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("from module import name1, name2 # comment", "\n", config) == "from module import(\n    name1,\n    name2, # comment)"

def test_line_wrap_with_noqa_in_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.GRID, comment_prefix="# ", include_trailing_comma=True, use_parentheses=True, indent="    ")
    assert line("from module import name1, name2 # noqa", "\n", config) == "from module import(\n    name1,\n    name2# noqa)"


# LLM-generated content at query #46
#--------------------------

```
def test_line_empty_content_after_split():
    content = ""
    line_separator = "\n"
    config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
        indent="    ",
    )
    assert not content


# LLM-generated content at query #47
#--------------------------

```python
def test_predicate_at_line_4_evaluates_to_false():
    content = "short_line"
    line_separator = "\n"
    config = DEFAULT_CONFIG
    config.line_length = 20
    config.multi_line_output = Modes.NOQA
    result = line(content, line_separator, config)
    assert "# NOQA" not in result


# LLM-generated content at query #48
#--------------------------

```python
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "import os"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_with_wrap_and_parentheses():
    config = Config(line_length=10, wrap_length=None, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "import os.path as osp"
    line_separator = "\n"
    result = line(content, line_separator, config)
    expected = "import os.path as (\n    osp\n)"
    assert result == expected

def test_line_with_wrap_and_backslash():
    config = Config(line_length=10, wrap_length=None, use_parentheses=False, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "import os.path as osp"
    line_separator = "\n"
    result = line(content, line_separator, config)
    expected = "import os.path as \\\n    osp"
    assert result == expected

def test_line_with_noqa():
    config = Config(line_length=10, wrap_length=None, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.NOQA)
    content = "import os.path as osp"
    line_separator = "\n"
    result = line(content, line_separator, config)
    expected = "import os.path as osp# NOQA"
    assert result == expected

def test_line_with_comment():
    config = Config(line_length=10, wrap_length=None, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "import os.path as osp # some comment"
    line_separator = "\n"
    result = line(content, line_separator, config)
    expected = "import os.path as (\n    osp# some comment\n)"
    assert result == expected

def test_line_with_comment_and_noqa():
    config = Config(line_length=10, wrap_length=None, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.NOQA)
    content = "import os.path as osp # some comment"
    line_separator = "\n"
    result = line(content, line_separator, config)
    expected = "import os.path as osp# some comment# NOQA"
    assert result == expected


# LLM-generated content at query #49
#--------------------------

```python
def test_line_no_wrap_needed():
    config = Config(line_length=100, wrap_length=80, multi_line_output=0, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrap_with_comment():
    config = Config(line_length=20, wrap_length=10, multi_line_output=1, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module import very_long_name  # some comment", "\n", config)
    assert result == "from module import(\n    very_long_name  # some comment\n)"

def test_line_wrap_noqa():
    config = Config(line_length=20, wrap_length=10, multi_line_output=3, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module import very_long_name", "\n", config)
    assert result == "from module import very_long_name#  NOQA"

def test_line_wrap_with_parentheses():
    config = Config(line_length=20, wrap_length=10, multi_line_output=2, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module import name1, name2, name3", "\n", config)
    assert result == "from module import(\n    name1,\n    name2,\n    name3,\n)"

def test_line_wrap_with_as_keyword():
    config = Config(line_length=20, wrap_length=10, multi_line_output=1, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module import very_long_name as short", "\n", config)
    assert result == "from module import very_long_name as short"

def test_line_wrap_with_trailing_comma():
    config = Config(line_length=20, wrap_length=10, multi_line_output=1, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module import name1, name2, name3  # noqa", "\n", config)
    assert result == "from module import(\n    name1,\n    name2,\n    name3#  noqa\n)"

def test_line_wrap_with_dot_separator():
    config = Config(line_length=20, wrap_length=10, multi_line_output=1, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("module.submodule.very_long_name", "\n", config)
    assert result == "module.submodule.very_long_name"

def test_line_wrap_with_cimport():
    config = Config(line_length=20, wrap_length=10, multi_line_output=1, indent="    ", comment_prefix="# ", include_trailing_comma=True, use_parentheses=True)
    result = line("from module cimport name1, name2, name3", "\n", config)
    assert result == "from module cimport(\n    name1,\n    name2,\n    name3,\n)"


# LLM-generated content at query #50
#--------------------------

```python
def test_line_with_no_wrapping_needed():
    content = "import os"
    line_separator = "\n"
    config = Config(line_length=100, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert line(content, line_separator, config) == "import os"

def test_line_with_wrapping_and_comment():
    content = "from module import very_long_function_name # noqa"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    expected = "from module import (\n    very_long_function_name # noqa\n)"
    assert line(content, line_separator, config) == expected

def test_line_with_wrapping_and_noqa():
    content = "from module import very_long_function_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.NOQA)
    assert line(content, line_separator, config) == "from module import very_long_function_name # NOQA"

def test_line_with_wrapping_and_as_clause():
    content = "from module import very_long_function_name as short_name"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    expected = "from module import (\n    very_long_function_name as short_name\n)"
    assert line(content, line_separator, config) == expected

def test_line_with_wrapping_and_trailing_comma():
    content = "from module import very_long_function_name,"
    line_separator = "\n"
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True)
    expected = "from module import (\n    very_long_function_name,\n)"
    assert line(content, line_separator, config) == expected


# LLM-generated content at query #51
#--------------------------

```python
def test_balanced_wrapping_predicate_evaluates_to_true():
    import_start = "from module import"
    from_imports = ["item1", "item2", "item3", "item4", "item5"]
    comments = ["# comment"]
    line_separator = "\n"
    config = Config(balanced_wrapping=True, wrap_length=20, line_length=20, indent="    ", comment_prefix="", ignore_comments=False, include_trailing_comma=True)
    multi_line_output = None
    explode = False
    statement = import_statement(import_start, from_imports, comments, line_separator, config, multi_line_output, explode)
    assert statement.count(line_separator) > 1


# LLM-generated content at query #52
#--------------------------

```python
def test_line_no_wrap_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix=" #", indent="    ")
    assert line("short line", "\n", config) == "short line"

def test_line_wrap_needed_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=False, include_trailing_comma=False, comment_prefix=" #", indent="    ")
    assert line("this line is too long", "\n", config) == "this line is too long # NOQA"

def test_line_wrap_needed_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" #", indent="    ")
    assert line("import very_long_module_name # this is a comment", "\n", config) == "import (very_long_module_name, # this is a comment\n    )"

def test_line_wrap_needed_with_noqa_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" #", indent="    ")
    assert line("import very_long_module_name # NOQA", "\n", config) == "import (very_long_module_name # NOQA\n    )"

def test_line_wrap_needed_with_as_keyword():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" #", indent="    ")
    assert line("import very_long_module_name as vlm", "\n", config) == "import very_long_module_name as vlm"

def test_line_wrap_needed_with_dot_keyword():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix=" #", indent="    ")
    assert line("from package import very_long_module_name", "\n", config) == "from package import (very_long_module_name\n    )"


# LLM-generated content at query #53
#--------------------------

```python
def test_line_no_wrapping_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.NOQA)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed_with_noqa():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA)
    result = line("import os.path", "\n", config)
    assert result == "import os.path# NOQA"

def test_line_wrapping_needed_with_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import os.path as osp", "\n", config)
    assert result == "import os.path as (\n    osp,\n)"

def test_line_wrapping_needed_without_parentheses():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=False, include_trailing_comma=True)
    result = line("import os.path as osp", "\n", config)
    assert result == "import os.path as \\\n    osp"

def test_line_wrapping_needed_with_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True)
    result = line("import os.path as osp  # noqa", "\n", config)
    assert result == "import os.path as (\n    osp  # noqa,\n)"

def test_line_wrapping_needed_with_noqa_comment():
    config = Config(line_length=10, wrap_length=None, multi_line_output=Modes.NOQA, use_parentheses=True, include_trailing_comma=True)
    result = line("import os.path as osp  # noqa", "\n", config)
    assert result == "import os.path as osp  # noqa"


# LLM-generated content at query #54
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    config = Config(line_length=100, wrap_length=80, use_parentheses=True, include_trailing_comma=True, comment_prefix="# ", indent="    ", multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    content = "a" * 101
    line_separator = "\n"
    line_parts = ["a" * 50, "b" * 50]
    assert (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts


# LLM-generated content at query #55
#--------------------------

```python
def test_predicate_at_line_15_evaluates_to_true():
    config = Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, use_parentheses=True, include_trailing_comma=True, comment_prefix="#", line_length=80, wrap_length=80, indent="    ")
    content = "import very_long_module_name as very_long_module_name_alias"
    line_separator = "\n"
    comment = "noqa: E501"
    line_without_comment = content
    line_parts = re.split(r"\bas\b", line_without_comment)
    assert comment and not (config.use_parentheses and "noqa" in comment)


# LLM-generated content at query #56
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short_line"
    config = Config(line_length=10, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    line_parts = ["part1", "part2"]
    result = (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts
    assert not result


# LLM-generated content at query #57
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    class Config:
        def __init__(self, wrap_length, line_length):
            self.wrap_length = wrap_length
            self.line_length = line_length

    config = Config(wrap_length=50, line_length=80)
    content = "a" * 100
    line_parts = ["part1", "part2", "part3"]
    assert (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts


# LLM-generated content at query #58
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short_content"
    config = Config(line_length=80, wrap_length=80)
    result = (len(content) + 2) > (config.wrap_length or config.line_length)
    assert not result


# LLM-generated content at query #59
#--------------------------

```python
def test_wrap_length_used_when_provided():
    class Config:
        def __init__(self):
            self.line_length = 100
            self.wrap_length = 50
            self.multi_line_output = 1
    config = Config()
    content = "a" * 60
    assert (len(content) + 2) > (config.wrap_length or config.line_length)

def test_line_length_used_when_wrap_length_not_provided():
    class Config:
        def __init__(self):
            self.line_length = 100
            self.wrap_length = None
            self.multi_line_output = 1
    config = Config()
    content = "a" * 99
    assert (len(content) + 2) > (config.wrap_length or config.line_length)


# LLM-generated content at query #60
#--------------------------

```python
def test_line_no_wrapping_needed():
    config = Config(line_length=100, multi_line_output=Modes.NOQA)
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed_with_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os # noqa", "\n", config)
    assert result == "import os # noqa"

def test_line_wrapping_needed_without_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os", "\n", config)
    assert result == "import os"

def test_line_wrapping_needed_with_splitter():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os.path", "\n", config)
    assert result == "import os.path"

def test_line_wrapping_needed_with_splitter_and_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os.path # noqa", "\n", config)
    assert result == "import os.path # noqa"

def test_line_wrapping_needed_with_splitter_and_long_content():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os.path as osp", "\n", config)
    assert result == "import os.path as osp"

def test_line_wrapping_needed_with_splitter_and_long_content_and_comment():
    config = Config(line_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, include_trailing_comma=True, use_parentheses=True, comment_prefix=" # ")
    result = line("import os.path as osp # noqa", "\n", config)
    assert result == "import os.path as osp # noqa"

def test_line_wrapping_needed_with_noqa_mode():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix=" # ")
    result = line("import os.path", "\n", config)
    assert result == "import os.path # NOQA"

def test_line_wrapping_needed_with_noqa_mode_and_comment():
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix=" # ")
    result = line("import os.path # noqa", "\n", config)
    assert result == "import os.path # noqa"


# LLM-generated content at query #61
#--------------------------

```
def test_use_parentheses_with_noqa_comment():
    config = Config(
        use_parentheses=True,
        comment_prefix="# ",
        include_trailing_comma=True,
        wrap_length=80,
        line_length=80,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    content = "from module import very_long_name_that_exceeds_line_length"
    line_separator = "\n"
    comment = "noqa"
    line_without_comment = content
    splitter = "import "
    line_parts = re.split(r"\b" + re.escape(splitter) + r"\b", line_without_comment)
    next_line = ["very_long_name_that_exceeds_line_length"]
    cont_line = _wrap_line(
        config.indent + splitter.join(next_line).lstrip(),
        line_separator,
        config,
    )
    assert config.use_parentheses


# LLM-generated content at query #62
#--------------------------

```python
def test_predicate_at_line_15_evaluates_to_True():
    content = "import module # some comment"
    line_without_comment = "import module"
    comment = "some comment"
    config = Config(use_parentheses=False)
    assert comment and not (config.use_parentheses and "noqa" in comment)


# LLM-generated content at query #63
#--------------------------

```
def test_include_trailing_comma_with_use_parentheses_and_no_trailing_comma():
    class Config:
        include_trailing_comma = True
        use_parentheses = True
        comment_prefix = " #"
    
    line_without_comment = "some content"
    config = Config()
    assert (
        ","
        if (
            config.include_trailing_comma
            and config.use_parentheses
            and not line_without_comment.rstrip().endswith(",")
        )
        else ""
    ) == ","


# LLM-generated content at query #64
#--------------------------

```python
def test_line_with_no_wrapping_needed():
    config = Config(line_length=80, wrap_length=None, multi_line_output=Modes.NOQA, comment_prefix="#", include_trailing_comma=False, use_parentheses=False, indent="    ")
    content = "import os"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import os"

def test_line_with_wrapping_needed():
    config = Config(line_length=20, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="#", include_trailing_comma=True, use_parentheses=True, indent="    ")
    content = "import very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import (\n    very_long_module_name_that_exceeds_line_length,\n)"

def test_line_with_comment():
    config = Config(line_length=20, wrap_length=10, multi_line_output=Modes.VERTICAL_HANGING_INDENT, comment_prefix="#", include_trailing_comma=True, use_parentheses=True, indent="    ")
    content = "import module # this is a comment"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import (\n    module, # this is a comment\n)"

def test_line_with_noqa():
    config = Config(line_length=20, wrap_length=10, multi_line_output=Modes.NOQA, comment_prefix="#", include_trailing_comma=False, use_parentheses=False, indent="    ")
    content = "import very_long_module_name_that_exceeds_line_length"
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "import very_long_module_name_that_exceeds_line_length# NOQA"


# LLM-generated content at query #65
#--------------------------

```python
def test_predicate_at_line_30_evaluates_to_true():
    class Config:
        def __init__(self, wrap_length, line_length):
            self.wrap_length = wrap_length
            self.line_length = line_length

    config = Config(wrap_length=50, line_length=80)
    content = "a" * 100
    line_parts = ["part1", "part2", "part3"]
    assert (len(content) + 2) > (config.wrap_length or config.line_length) and line_parts


# LLM-generated content at query #66
#--------------------------

```python
def test_predicate_at_line_42_evaluates_to_true():
    class Config:
        def __init__(self, use_parentheses):
            self.use_parentheses = use_parentheses

    config = Config(use_parentheses=True)
    assert config.use_parentheses


# LLM-generated content at query #67
#--------------------------

```python
def test_predicate_evaluates_to_true():
    config = Config(include_trailing_comma=True, use_parentheses=True)
    content = "example_content"
    assert line(content, "\n", config) == content


# LLM-generated content at query #68
#--------------------------

```python
def test_line_predicate_evaluates_to_false():
    content = "short_line"
    line_separator = "\n"
    class Config:
        multi_line_output = "NOQA"
        line_length = 20
        wrap_length = None
        indent = "    "
        use_parentheses = False
        include_trailing_comma = False
        comment_prefix = "# "
    config = Config()
    assert not (len(content) + 2 > (config.wrap_length or config.line_length))


# LLM-generated content at query #69
#--------------------------

```python
def test_predicate_at_line_11_evaluates_to_true():
    content = "from module import something"
    line_without_comment = content
    splitter = "import "
    exp = r"\b" + re.escape(splitter) + r"\b"
    assert re.search(exp, line_without_comment) and not line_without_comment.strip().startswith(splitter)


# LLM-generated content at query #70
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short_line"
    config = Config(line_length=100, wrap_length=None)
    line_separator = "\n"
    result = line(content, line_separator, config)
    assert result == "short_line"


# LLM-generated content at query #71
#--------------------------

```python
def test_predicate_at_line_29_evaluates_to_false():
    content = "short_line"
    line_parts = []
    config = Config(
        line_length=10,
        wrap_length=10,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# ",
    )
    assert not ((len(content) + 2) > (config.wrap_length or config.line_length) and line_parts)


