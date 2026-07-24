####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    long_line = "a" * 100
    result = line(long_line, "\n", config)
    assert result == long_line

    # Test line with import splitter
    content = "from very_long_module_name import very_long_function_name"
    config = Config(line_length=40, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert result.startswith("from very_long_module_name import")
    assert "very_long_function_name" in result

    # Test line with dot splitter
    content = "module.submodule.very_long_attribute_name"
    config = Config(line_length=30, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "module.submodule" in result

    # Test line with as splitter
    content = "import very_long_module_name as vlm"
    config = Config(line_length=30, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "very_long_module_name" in result

    # Test line with comment
    content = "import os  # comment"
    config = Config(line_length=20, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "# comment" in result

    # Test NOQA mode
    content = "import os"
    config = Config(line_length=5, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert "NOQA" in result

    # Test with parentheses and trailing comma
    content = "from module import function"
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line(content, "\n", config)
    assert "(" in result
    assert ")" in result
    assert "," in result

    # Test with noqa comment and parentheses
    content = "from module import function  # noqa"
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line(content, "\n", config)
    assert "# noqa" in result
    assert "(" in result
    assert ")" in result

    # Test line that starts with splitter (should not split)
    content = "import os"
    config = Config(line_length=5, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\\n" not in result

    # Test with cimport splitter
    content = "from cython_module cimport function"
    config = Config(line_length=30, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "cimport" in result


# LLM-generated content at query #2
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom config
    config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very.long.module.name", ["very_long_import_name1", "very_long_import_name2"], config=config)
    assert "(" in result and ")" in result

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Should be single line

    # Test with multi_line_output override
    result = import_statement("from module", ["import1", "import2", "import3", "import4"], 
                            multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert "(" in result and ")" in result

    # Test with many imports that should wrap
    many_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from module", many_imports, config=config)
    assert result.count("\n") > 0

    # Test balanced_wrapping
    config_balanced = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement("from module", ["very_long_import_name" * 3, "another_long_import" * 3], 
                            config=config_balanced)
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) - 1

    # Test with remove_comments=True
    config_no_comments = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.GRID,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2"], 
                            comments=["comment1", "comment2"], config=config_no_comments)
    assert "comment1" not in result and "comment2" not in result

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"


# LLM-generated content at query #3
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test explode mode
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result.count("\n") == 2  # Should have multiple lines
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=custom_config)
    assert "    " in result  # Should have indentation
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or "import1, import2" in result
    
    # Test balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=50,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True
    )
    long_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from very_long_module_name", long_imports, config=balanced_config)
    assert isinstance(result, str)
    
    # Test with multi_line_output override
    result = import_statement(
        "from module", 
        ["import1", "import2", "import3"], 
        multi_line_output=Modes.VERTICAL_GRID_GROUPED
    )
    assert isinstance(result, str)
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module"
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with trailing comma config
    comma_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=comma_config)
    if result.count("\n") > 0:
        assert result.strip().endswith(",") or "import3" in result
    
    # Test without trailing comma config
    no_comma_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=no_comma_config)
    if result.count("\n") > 0:
        assert not result.strip().endswith(",")


# LLM-generated content at query #4
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line splitting on 'import' with parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "from module import (" in result
    assert "very_long_name1," in result
    assert "very_long_name2," in result
    assert result.endswith(")")

    # Test line splitting on 'as' with parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # "
    )
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "import very_long_module_name as" in result
    assert "very_long_alias" in result
    assert "\\" not in result

    # Test line splitting on '.' with parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from package.subpackage import very_long_name", "\n", config)
    assert "from package.subpackage import (" in result
    assert "very_long_name," in result
    assert result.endswith(")")

    # Test line with comment preservation
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # "
    )
    result = line("from module import name1, name2  # some comment", "\n", config)
    assert "# some comment" in result
    assert result.endswith(")")

    # Test line with noqa comment and parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # "
    )
    result = line("from module import name1, name2  # noqa", "\n", config)
    assert "# noqa" in result
    assert result.endswith(")")

    # Test line splitting without parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
        indent="    ",
        comment_prefix="  # "
    )
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with trailing comma logic
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import name1, name2  # comment", "\n", config)
    assert "name2," not in result
    assert result.endswith(")")

    # Test very short line length with splitter
    config = Config(
        line_length=15,
        wrap_length=15,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("import module.submodule", "\n", config)
    assert "import (" in result
    assert "module.submodule," in result
    assert result.endswith(")")


# LLM-generated content at query #5
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.GRID
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "import " + "very_long_module_name_" * 5
    result = line(long_line, "\n", config)
    assert result.endswith("# NOQA")
    assert "# NOQA" in result

    # Test line with comment and NOQA mode
    line_with_comment = "import os  # some comment"
    result = line(line_with_comment, "\n", config)
    assert result == line_with_comment

    # Test line that needs wrapping with 'import' splitter
    config.multi_line_output = Modes.GRID
    config.line_length = 20
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.indent = "    "
    long_import = "from very.long.package.name import very_long_module_name"
    result = line(long_import, "\n", config)
    assert "\\" in result or "(" in result

    # Test line with 'as' splitter
    config.line_length = 30
    import_as = "import very_long_module_name as vlm"
    result = line(import_as, "\n", config)
    assert "as" in result

    # Test line with '.' splitter
    config.line_length = 25
    dotted_import = "from module.submodule.anothersub import something"
    result = line(dotted_import, "\n", config)
    assert "." in result or "import" in result

    # Test line with comment and parentheses
    config.line_length = 40
    line_with_complex_comment = "import os  # noqa: F401"
    config.use_parentheses = True
    result = line(line_with_complex_comment, "\n", config)
    assert "noqa" in result

    # Test line without parentheses
    config.use_parentheses = False
    config.line_length = 25
    long_line_no_parens = "import very_long_module_name_here"
    result = line(long_line_no_parens, "\n", config)
    assert "\\" in result

    # Test line with trailing comma handling
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.line_length = 35
    line_with_comma = "from module import item1, item2, item3"
    result = line(line_with_comma, "\n", config)
    assert "," in result

    # Test vertical hanging indent mode
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 30
    vertical_line = "from module import item1, item2, item3, item4"
    result = line(vertical_line, "\n", config)
    assert "\n" in result

    # Test vertical grid grouped mode
    config.multi_line_output = Modes.VERTICAL_GRID_GROUPED
    result = line(vertical_line, "\n", config)
    assert "\n" in result

    # Test line that's exactly at length limit
    config.line_length = 20
    exact_length_line = "import os" + " " * 12
    result = line(exact_length_line, "\n", config)
    assert len(result.strip()) <= 20 or "\\" in result or "\n" in result

    # Test line with cimport splitter
    config.line_length = 25
    cimport_line = "cimport numpy as np"
    result = line(cimport_line, "\n", config)
    assert "cimport" in result

    # Test comment prefix handling
    config.comment_prefix = "  # "
    line_with_custom_comment = "import os  # comment"
    result = line(line_with_custom_comment, "\n", config)
    assert config.comment_prefix in result

    # Test wrap_length different from line_length
    config.line_length = 80
    config.wrap_length = 40
    long_line_wrap = "import " + "x" * 50
    result = line(long_line_wrap, "\n", config)
    assert len(result) > 0


# LLM-generated content at query #6
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line wrapping with import splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import (" in result
    assert "very_long_name," in result
    assert "\n" in result

    # Test line wrapping with as splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as" in result
    assert "vlm" in result

    # Test line with comment handling
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name  # some comment", "\n", config)
    assert "  # some comment" in result

    # Test line with noqa comment in parentheses mode
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "  # noqa" in result
    assert result.endswith(")")

    # Test line with dot splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("module.submodule.very_long_name", "\n", config)
    assert "module.submodule." in result
    assert "very_long_name" in result

    # Test line without parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=False,
        include_trailing_comma=False,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line that starts with splitter (should not split)
    config = Config(
        line_length=10,
        wrap_length=10,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line with trailing comma handling
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2  # comment", "\n", config)
    assert "," in result
    assert "  # comment" in result

    # Test vertical grid grouped mode
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import (" in result
    assert "\n" in result


# LLM-generated content at query #7
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import os  # NOQA", "\n", config)
    assert result == "import os  # NOQA"

    # Test line with comment and wrapping
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import very_long_module  # some comment", "\n", config)
    assert "very_long_module" in result
    assert "  # some comment" in result

    # Test line with 'import' splitter
    config = Config(line_length=30, multi_line_output=Modes.VERTICAL_HANGING_INDENT, 
                    use_parentheses=True, include_trailing_comma=True, indent="    ")
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "from module import" in result
    assert "very_long_name1" in result
    assert "very_long_name2" in result

    # Test line with 'as' splitter
    config = Config(line_length=25, multi_line_output=Modes.GRID, 
                    use_parentheses=True, indent="    ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "very_long_module_name" in result
    assert "as vlm" in result

    # Test line with '.' splitter
    config = Config(line_length=25, multi_line_output=Modes.GRID, 
                    use_parentheses=True, indent="    ")
    result = line("from package.subpackage import module", "\n", config)
    assert "package.subpackage" in result
    assert "import module" in result

    # Test line with trailing comma handling
    config = Config(line_length=35, multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                    use_parentheses=True, include_trailing_comma=True, indent="    ")
    result = line("from module import name1, name2, name3", "\n", config)
    assert "," in result
    assert ")" in result

    # Test line without parentheses
    config = Config(line_length=25, multi_line_output=Modes.GRID,
                    use_parentheses=False, indent="    ")
    result = line("import very_long_module_name", "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with cimport splitter
    config = Config(line_length=25, multi_line_output=Modes.GRID,
                    use_parentheses=True, indent="    ")
    result = line("from libc cimport math", "\n", config)
    assert "cimport" in result
    assert "math" in result

    # Test line with comment and noqa
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                    use_parentheses=True, comment_prefix="  #", include_trailing_comma=True)
    result = line("import module  # noqa: F401", "\n", config)
    assert "  # noqa: F401" in result
    assert ")" in result

    # Test line with wrap_length different from line_length
    config = Config(line_length=80, wrap_length=40, multi_line_output=Modes.GRID,
                    use_parentheses=True, indent="    ")
    result = line("from module import very_long_name1, very_long_name2, very_long_name3", "\n", config)
    assert len(result.split("\n")) > 1

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os"


# LLM-generated content at query #8
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        balanced_wrapping=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", ["import1", "import2", "import3"], config=custom_config)
    assert "(" in result and ")" in result

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    assert result.count("\n") > 0

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result

    # Test with balanced_wrapping
    config_with_balance = Config(balanced_wrapping=True, wrap_length=30, line_length=50)
    result = import_statement(
        "from module",
        ["very_long_import_name1", "very_long_import_name2", "very_long_import_name3"],
        config=config_with_balance,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 30 for line in lines[:-1])

    # Test single import with explode
    result = import_statement("from module", ["single_import"], explode=True)
    assert result == "from module import (\n    single_import,\n)"

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with ignore_comments
    config_no_comments = Config(ignore_comments=True)
    result = import_statement("from module", ["import1"], comments=["comment"], config=config_no_comments)
    assert "comment" not in result

    # Test that dynamic_indent is calculated correctly
    result = import_statement("from very_long_module", ["import1", "import2"], explode=True)
    assert result.startswith("from very_long_module import (")

    # Test trailing comma behavior
    config_no_trailing_comma = Config(include_trailing_comma=False, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement(
        "from module",
        ["import1", "import2", "import3"],
        config=config_no_trailing_comma,
    )
    if "(" in result and ")" in result:
        assert result.strip().endswith(")")

    # Test with remove_comments=False but no comments
    result = import_statement("from module", ["import1", "import2"], config=Config(ignore_comments=False))
    assert "import1" in result and "import2" in result


# LLM-generated content at query #9
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", ["import1", "import2", "import3"], config=custom_config)
    assert "import1" in result and "import2" in result and "import3" in result

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Should be single line

    # Test with multi_line_output override
    result = import_statement("from module", ["import1", "import2", "import3", "import4"], 
                            multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert result.count("\n") > 0

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement("from module", 
                            ["very_long_import_name_1", "very_long_import_name_2", 
                             "very_long_import_name_3", "very_long_import_name_4"],
                            config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) - 1

    # Test with remove_comments
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.GRID,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2"], 
                            comments=["comment1", "comment2"], config=no_comments_config)
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #10
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result
    assert "comment2" in result
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert "\n" in result
    assert "import1" in result
    assert "import2" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=45,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        balanced_wrapping=False,
        ignore_comments=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        wrap_length_or_line_length=45
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2", "very_long_import3"],
                             config=custom_config)
    assert "\n" in result
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert "\n" in result
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator
    
    # Test with many imports to force wrapping
    many_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from module", many_imports)
    # Result might be single or multi-line depending on DEFAULT_CONFIG
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=40,
        indent="  ",
        comment_prefix="# ",
        include_trailing_comma=False,
        balanced_wrapping=True,
        ignore_comments=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        wrap_length_or_line_length=40
    )
    result = import_statement("from module", 
                             [f"import{i}" for i in range(8)],
                             config=balanced_config)
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test with single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test that comments are properly integrated
    result = import_statement("from module", ["func1", "func2"], 
                             comments=["First function", "Second function"])
    assert "# First function" in result or "First function" in result


# LLM-generated content at query #11
#--------------------------

```python
def test_import_statement():
    from .settings import DEFAULT_CONFIG, Config
    from .wrap_modes import WrapModes as Modes

    # Test basic single line import
    result = import_statement(
        import_start="from module",
        from_imports=["function1", "function2"],
        config=DEFAULT_CONFIG,
    )
    assert result == "from module import function1, function2"

    # Test multi-line with wrap
    config = Config(
        line_length=40,
        wrap_length=40,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        include_trailing_comma=False,
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        import_start="from very_long_module_name",
        from_imports=["item1", "item2", "item3", "item4", "item5"],
        config=config,
    )
    assert "from very_long_module_name import (" in result
    assert "item1" in result
    assert "item5" in result

    # Test with comments
    result = import_statement(
        import_start="from module",
        from_imports=["func1", "func2"],
        comments=["comment1", "comment2"],
        config=config,
    )
    assert "comment1" in result or "comment2" in result

    # Test explode mode
    result = import_statement(
        import_start="from module",
        from_imports=["a", "b", "c"],
        explode=True,
    )
    lines = result.split("\n")
    assert len(lines) > 3
    assert all("import" in line for line in lines if line.strip())

    # Test balanced wrapping
    config_balanced = Config(
        line_length=50,
        wrap_length=50,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        include_trailing_comma=True,
        balanced_wrapping=True,
        comment_prefix="  # ",
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement(
        import_start="from module",
        from_imports=[f"item{i}" for i in range(10)],
        config=config_balanced,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        lengths = [len(line) for line in lines[:-1]]
        assert max(lengths) - min(lengths) < 10

    # Test with custom line separator
    result = import_statement(
        import_start="from module",
        from_imports=["x", "y", "z"],
        line_separator="\r\n",
        config=config,
    )
    assert "\r\n" in result

    # Test trailing comma
    config_comma = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        include_trailing_comma=True,
        balanced_wrapping=False,
        comment_prefix="  # ",
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement(
        import_start="from module",
        from_imports=["a", "b", "c", "d"],
        config=config_comma,
    )
    assert result.strip().endswith(",")

    # Test single import stays on one line
    result = import_statement(
        import_start="from module",
        from_imports=["single_function"],
        config=DEFAULT_CONFIG,
    )
    assert result.count("\n") == 0
    assert result == "from module import single_function"

    # Test with multi_line_output override
    result = import_statement(
        import_start="from module",
        from_imports=["a", "b", "c", "d", "e"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        config=DEFAULT_CONFIG,
    )
    assert "import (" in result or "import \\" in result


# LLM-generated content at query #12
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2", "import3"], explode=True)
    lines = result.split("\n")
    assert len(lines) > 1
    assert lines[0] == "from module import ("
    assert lines[-1] == ")"
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        balanced_wrapping=False,
        ignore_comments=False,
        comment_prefix=" # ",
        indent="    ",
        use_parentheses=True
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2", "very_long_import3"],
                             config=custom_config)
    assert "    " in result  # Should have indentation
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator
    
    # Test with many imports forcing multi-line
    many_imports = [f"import{i}" for i in range(20)]
    result = import_statement("from module", many_imports, config=custom_config)
    assert result.count("\n") > 0
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        balanced_wrapping=True,
        ignore_comments=False,
        comment_prefix=" # ",
        indent="    ",
        use_parentheses=True
    )
    result = import_statement("from module", 
                             ["very_long_import_name1", "very_long_import_name2", 
                              "very_long_import_name3", "very_long_import_name4"],
                             config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        # Check that lines are somewhat balanced
        lengths = [len(line) for line in lines[:-1]]
        assert max(lengths) - min(lengths) < 20
    
    # Test with remove_comments
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        balanced_wrapping=False,
        ignore_comments=True,
        comment_prefix=" # ",
        indent="    ",
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2"], 
                             comments=["comment1", "comment2"],
                             config=no_comments_config)
    assert "comment1" not in result and "comment2" not in result
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert result.count("\n") > 0
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"


# LLM-generated content at query #13
#--------------------------

```python
def test_line():
    from .settings import Config
    from .wrap_modes import WrapModes as Modes

    config = Config()
    line_separator = "\n"

    # Test 1: Simple line within length limit
    content = "import os"
    result = line(content, line_separator, config)
    assert result == "import os"

    # Test 2: Line exceeding length without wrap mode
    config.line_length = 10
    content = "import verylongmodulename"
    result = line(content, line_separator, config)
    assert result == "import verylongmodulename"

    # Test 3: Line with NOQA wrap mode
    config.multi_line_output = Modes.NOQA
    content = "import verylongmodulename"
    result = line(content, line_separator, config)
    assert result == "import verylongmodulename# NOQA"

    # Test 4: Line with comment and NOQA already present
    content = "import module  # NOQA"
    result = line(content, line_separator, config)
    assert result == "import module  # NOQA"

    # Test 5: Line with import splitter
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 20
    content = "from very.long.package.name import module"
    result = line(content, line_separator, config)
    assert "\\" in result or "(" in result

    # Test 6: Line with as splitter
    content = "import verylongmodulename as vlm"
    result = line(content, line_separator, config)
    assert "as" in result

    # Test 7: Line with cimport splitter
    content = "from libc cimport verylongfunctionname"
    result = line(content, line_separator, config)
    assert "cimport" in result

    # Test 8: Line with dot splitter
    content = "very.long.package.name.module"
    result = line(content, line_separator, config)
    assert "." in result

    # Test 9: Line with comment handling
    config.use_parentheses = True
    content = "from package import module  # some comment"
    result = line(content, line_separator, config)
    assert "#" in result

    # Test 10: Line with trailing comma config
    config.include_trailing_comma = True
    content = "from package import module1, module2"
    result = line(content, line_separator, config)
    assert result.count(",") >= 2

    # Test 11: Line starting with splitter (should not split)
    content = "import module"
    result = line(content, line_separator, config)
    assert result == "import module"

    # Test 12: Empty line
    content = ""
    result = line(content, line_separator, config)
    assert result == ""

    # Test 13: Line exactly at length limit
    config.line_length = 11
    content = "import os"
    result = line(content, line_separator, config)
    assert result == "import os"

    # Test 14: Line with noqa in comment and use_parentheses
    config.ignore_comments = False
    content = "from package import module  # noqa"
    result = line(content, line_separator, config)
    assert "noqa" in result

    # Test 15: Line with comment prefix in output
    config.comment_prefix = "# "
    content = "from package import module  # comment"
    result = line(content, line_separator, config)
    assert "# " in result


# LLM-generated content at query #14
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line wrapping with import splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from very.long.package.name import something", "\n", config)
    assert "\\\n" in result
    assert "    " in result

    # Test line wrapping with as splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "\\\n" in result
    assert "as" in result

    # Test line wrapping with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        indent="    ",
    )
    result = line("from module import very_long_name", "\n", config)
    assert "(" in result
    assert ")" in result
    assert "," in result

    # Test line with comment handling
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import module  # some comment", "\n", config)
    assert "# some comment" in result

    # Test line with noqa comment and parentheses
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        comment_prefix="  # ",
    )
    result = line("import module  # noqa", "\n", config)
    assert "# noqa" in result
    assert ")" in result

    # Test line wrapping with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("cimport very.long.cython.module", "\n", config)
    assert "\\\n" in result

    # Test line wrapping with dot splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("module.submodule.very_long_attribute", "\n", config)
    assert "\\\n" in result

    # Test line that exactly matches line length
    config = Config(line_length=20, multi_line_output=Modes.GRID)
    result = line("import module1234567", "\n", config)
    assert result == "import module1234567"

    # Test line with wrap_length different from line_length
    config = Config(line_length=80, wrap_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_function_name", "\n", config)
    assert len(result.split("\n")[0]) <= 20

    # Test line with comment prefix in last line ending with parenthesis
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        comment_prefix="  # ",
    )
    result = line("import module  # comment", "\n", config)
    assert ")" in result
    assert "# comment" in result
    assert not result.endswith("# comment)")

    # Test line with trailing comma and comment
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line("import module  # comment", "\n", config)
    assert "," in result
    assert "# comment" in result


# LLM-generated content at query #15
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with NOQA mode but already has NOQA comment
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line with import splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_name" in result

    # Test line with as splitter
    result = line("import module as very_long_alias_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_alias_name" in result

    # Test line with dot splitter
    result = line("module.very_long_attribute_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_attribute_name" in result

    # Test line with cimport splitter
    result = line("cimport module.very_long_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_name" in result

    # Test line with comment
    config = Config(line_length=25, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("from module import name  # comment", "\n", config)
    assert "  # comment" in result

    # Test line with use_parentheses and include_trailing_comma
    config = Config(
        line_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment_prefix="  #",
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "(\n" in result
    assert "    very_long_name,\n)" in result or "    very_long_name,\n)" in result

    # Test line with noqa comment and use_parentheses
    config = Config(
        line_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment_prefix="  #",
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "  # noqa" in result
    assert result.endswith(")")

    # Test line with comment and use_parentheses but no noqa
    config = Config(
        line_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment_prefix="  #",
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name  # regular comment", "\n", config)
    assert "  # regular comment" in result
    assert "," in result  # Should have trailing comma

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("import module", "\n", config)
    assert "\\\n" not in result

    # Test with wrap_length different from line_length
    config = Config(line_length=80, wrap_length=20, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert "\\\n" in result
    assert len(result.split("\n")[0]) <= 20

    # Test with vertical grid grouped mode
    config = Config(
        line_length=25,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        comment_prefix="  #",
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "(\n" in result
    assert "\n)" in result

    # Test line with multiple splitters
    config = Config(line_length=30, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("import module.submodule as alias", "\n", config)
    assert "\\\n" in result


# LLM-generated content at query #16
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["item1", "item2"])
    assert result == "from module import item1, item2"

    # Test with explode=True creates multi-line
    result = import_statement("from module", ["item1", "item2"], explode=True)
    assert result == "from module import (\n    item1,\n    item2,\n)"

    # Test with comments
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"])
    assert "# comment1" in result and "# comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very.long.module.name", ["item1", "item2", "item3"], config=custom_config)
    assert "from very.long.module.name import (" in result

    # Test with many imports forcing wrap
    many_imports = [f"item{i}" for i in range(10)]
    result = import_statement("from module", many_imports, config=Config(line_length=40))
    assert "\n" in result  # Should wrap to multiple lines

    # Test with line_separator
    result = import_statement("from module", ["item1", "item2"], line_separator="\r\n", explode=True)
    assert "\r\n" in result

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["item1", "item2", "item3"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        config=Config(line_length=30),
    )
    assert "import (" in result

    # Test balanced_wrapping
    config_with_balance = Config(balanced_wrapping=True, line_length=50)
    result = import_statement(
        "from module",
        [f"very_long_item_name_{i}" for i in range(5)],
        config=config_with_balance,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 50 for line in lines)

    # Test single line with trailing comma config
    config_with_comma = Config(include_trailing_comma=True, use_parentheses=True, line_length=100)
    result = import_statement("from module", ["item1", "item2"], config=config_with_comma)
    assert result.endswith(",)") or "item1, item2)" in result

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with ignore_comments
    config_ignore_comments = Config(ignore_comments=True)
    result = import_statement(
        "from module",
        ["item1", "item2"],
        comments=["comment1", "comment2"],
        config=config_ignore_comments,
    )
    assert "# comment1" not in result and "# comment2" not in result


# LLM-generated content at query #17
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config()
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has NOQA mode
    config.multi_line_output = Modes.NOQA
    config.line_length = 10
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line wrapping with 'import' splitter
    config.multi_line_output = Modes.GRID
    config.line_length = 20
    config.wrap_length = 20
    config.indent = "    "
    result = line("from module import very_long_name", "\n", config)
    assert "\\" in result or "(" in result

    # Test line wrapping with 'as' splitter
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "as" in result

    # Test line wrapping with '.' splitter
    result = line("from package.subpackage import name", "\n", config)
    assert "." in result

    # Test line with comment handling
    config.use_parentheses = True
    config.comment_prefix = "  # "
    result = line("import module  # some comment", "\n", config)
    assert "some comment" in result

    # Test line with trailing comma when use_parentheses is True
    config.include_trailing_comma = True
    result = line("from mod import long_name", "\n", config)
    assert result.endswith(",") or ")" in result

    # Test vertical hanging indent mode
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    result = line("from module import name1, name2, name3", "\n", config)
    assert "(" in result and ")" in result

    # Test vertical grid grouped mode
    config.multi_line_output = Modes.VERTICAL_GRID_GROUPED
    result = line("from module import multiple_names_here", "\n", config)
    assert "(" in result and ")" in result

    # Test line that exactly matches line length
    config.line_length = 15
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test with custom line separator
    result = line("import very_long_module", "\r\n", config)
    assert "\r\n" in result or result == "import very_long_module"

    # Test with comment containing noqa
    config.multi_line_output = Modes.GRID
    result = line("import module  # noqa", "\n", config)
    assert "noqa" in result

    # Test empty line
    result = line("", "\n", config)
    assert result == ""


# LLM-generated content at query #18
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.GRID
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line that exceeds length with NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "import " + "very_long_module_name_" * 10
    result = line(long_line, "\n", config)
    assert result.endswith("# NOQA")

    # Test line with comment that doesn't trigger wrapping
    config.multi_line_output = Modes.GRID
    content = "import module  # some comment"
    result = line(content, "\n", config)
    assert result == content

    # Test line that needs wrapping with 'import' splitter
    config.line_length = 30
    config.wrap_length = 30
    config.indent = "    "
    content = "from very.long.package.name import very_long_module_name"
    result = line(content, "\n", config)
    assert "\\\n" in result or "(\n" in result

    # Test line that needs wrapping with 'as' splitter
    content = "import very_long_module_name as very_long_alias_name"
    result = line(content, "\n", config)
    assert "as" in result

    # Test line that needs wrapping with '.' splitter
    content = "from module.submodule.anothersubmodule import something"
    result = line(content, "\n", config)
    assert "." in result

    # Test with use_parentheses=True
    config.use_parentheses = True
    config.include_trailing_comma = True
    content = "from module import very_long_name_that_exceeds_line_length"
    result = line(content, "\n", config)
    assert result.startswith("from module import (")
    assert result.endswith(")")

    # Test with comment and noqa
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    content = "import module  # noqa"
    result = line(content, "\n", config)
    assert "noqa" in result

    # Test line exactly at length limit
    config.line_length = 20
    content = "import module"  # 13 chars
    result = line(content, "\n", config)
    assert result == content

    # Test with trailing comma handling
    config.use_parentheses = True
    config.include_trailing_comma = True
    content = "from module import name"
    result = line(content, "\n", config)
    assert "," in result or ")" in result

    # Test with comment prefix
    config.comment_prefix = "  # "
    content = "import module  # comment"
    result = line(content, "\n", config)
    assert config.comment_prefix in result


# LLM-generated content at query #19
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["function1", "function2"])
    assert result == "from module import function1, function2"

    # Test with explode=True
    result = import_statement("from module", ["func1", "func2", "func3"], explode=True)
    assert result == "from module import (\n    func1,\n    func2,\n    func3,\n)"

    # Test with comments
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        ignore_comments=False,
        comment_prefix=" # ",
        use_parentheses=True,
        wrap_length_or_line_length=40
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_function_name1", "very_long_function_name2"], 
                             config=custom_config)
    assert "very_long_module_name" in result
    assert "very_long_function_name" in result

    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["item1", "item2", "item3", "item4", "item5"], 
                             multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert "item1" in result and "item5" in result

    # Test with line_separator
    result = import_statement("from module", ["func1", "func2"], line_separator="\r\n")
    assert "\r\n" not in result or result.count("\r\n") == 0  # Single line

    # Test balanced_wrapping
    balanced_config = Config(
        line_length=80,
        wrap_length=30,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix=" # ",
        ignore_comments=False,
        use_parentheses=True,
        wrap_length_or_line_length=30
    )
    result = import_statement("from module", 
                             ["very_long_name1", "very_long_name2", "very_long_name3"], 
                             config=balanced_config)
    assert "very_long_name1" in result

    # Test with empty from_imports
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with single import
    result = import_statement("from module", ["single_function"])
    assert result == "from module import single_function"

    # Test with remove_comments
    no_comments_config = Config(
        line_length=80,
        wrap_length=80,
        ignore_comments=True,
        comment_prefix=" # ",
        indent="    ",
        include_trailing_comma=False,
        multi_line_output=Modes.GRID,
        balanced_wrapping=False,
        use_parentheses=True,
        wrap_length_or_line_length=80
    )
    result = import_statement("from module", ["func1", "func2"], 
                             comments=["comment1", "comment2"], 
                             config=no_comments_config)
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #20
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert result == "from module import import1, import2  # comment1  # comment2"
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        ignore_comments=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2"], 
                             config=custom_config)
    assert "from very_long_module_name import (" in result
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or result == "from module import import1, import2"
    
    # Test with multi_line_output override
    result = import_statement("from module", ["import1", "import2", "import3", "import4"], 
                             multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert result.count("\n") > 0
    
    # Test with balanced_wrapping
    config_with_balance = Config(
        line_length=50,
        wrap_length=30,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"], 
                             config=config_with_balance)
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 50 for line in lines)
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with remove_comments=True
    config_no_comments = Config(ignore_comments=True)
    result = import_statement("from module", ["import1", "import2"], 
                             comments=["comment1", "comment2"], 
                             config=config_no_comments)
    assert "#" not in result
    
    # Test that explode forces vertical_hanging_indent regardless of config
    config_other_mode = Config(multi_line_output=Modes.GRID)
    result = import_statement("from module", ["import1", "import2"], 
                             explode=True, 
                             config=config_other_mode)
    assert result == "from module import (\n    import1,\n    import2,\n)"


# LLM-generated content at query #21
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    long_line = "a" * 100
    result = line(long_line, "\n", config)
    assert result == long_line

    # Test line with import splitter that needs wrapping
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    content = "from very.long.module.path import something"
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert result.startswith("from very.long.module.path import")
    assert "    something" in result

    # Test line with as splitter
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    content = "import very_long_module_name as vlm"
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "as" in result

    # Test line with dot splitter
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    content = "module.submodule.anothersubmodule.function"
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "." in result

    # Test line with comment
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ", comment_prefix="# ")
    content = "from module import something  # some comment"
    result = line(content, "\n", config)
    assert "# some comment" in result

    # Test NOQA mode
    config = Config(line_length=30, multi_line_output=Modes.NOQA, indent="    ", comment_prefix="# ")
    content = "a" * 100
    result = line(content, "\n", config)
    assert result.endswith("# NOQA")

    # Test line that already has NOQA comment
    content = "import something  # NOQA"
    result = line(content, "\n", config)
    assert result == content

    # Test with parentheses and trailing comma
    config = Config(
        line_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# "
    )
    content = "from module import very_long_import_name"
    result = line(content, "\n", config)
    assert "(" in result
    assert ")" in result
    assert ",\n" in result or result.endswith(",")

    # Test with comment and parentheses
    config = Config(
        line_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# "
    )
    content = "from module import name  # some comment"
    result = line(content, "\n", config)
    assert "# some comment" in result
    assert "(" in result
    assert ")" in result

    # Test with noqa comment and parentheses
    config = Config(
        line_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="# "
    )
    content = "from module import name  # noqa"
    result = line(content, "\n", config)
    assert "# noqa" in result
    assert "(" in result
    assert ")" in result

    # Test line starting with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    content = "import module"
    result = line(content, "\n", config)
    assert result == content

    # Test cimport splitter
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    content = "from libc.stdlib cimport malloc, free"
    result = line(content, "\n", config)
    assert "cimport" in result
    assert "\\\n" in result

    # Test with custom line separator
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    content = "from module import something"
    result = line(content, "\r\n", config)
    assert "\\\r\n" in result

    # Test edge case where content becomes empty after splitting
    config = Config(line_length=5, multi_line_output=Modes.GRID, indent="    ")
    content = "import a"
    result = line(content, "\n", config)
    assert result == content


# LLM-generated content at query #22
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    config = Config(line_length=10, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import verylongmodulename", "\n", config)
    assert result == "import verylongmodulename"

    # Test line with import splitter that needs wrapping
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("from module import verylongname", "\n", config)
    assert "\\\n" in result
    assert "verylongname" in result

    # Test line with comment and wrapping
    config = Config(line_length=30, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("from module import name1, name2  # comment", "\n", config)
    assert "comment" in result

    # Test NOQA mode when line exceeds length
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import verylong", "\n", config)
    assert result == "import verylong  # NOQA"

    # Test NOQA mode when line already has NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import x  # NOQA", "\n", config)
    assert result == "import x  # NOQA"

    # Test with use_parentheses and include_trailing_comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
        indent="    "
    )
    result = line("from module import verylongname", "\n", config)
    assert "(" in result
    assert ")" in result
    assert "," in result

    # Test with comment containing noqa
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
        indent="    "
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "noqa" in result
    assert ")" in result

    # Test with as splitter
    config = Config(line_length=15, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import module as mod", "\n", config)
    assert "\\\n" in result
    assert "mod" in result

    # Test with cimport splitter
    config = Config(line_length=15, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("cimport numpy as np", "\n", config)
    assert "\\\n" in result
    assert "np" in result

    # Test with dot splitter
    config = Config(line_length=15, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("module.submodule.function", "\n", config)
    assert "\\\n" in result
    assert "function" in result

    # Test line starting with splitter (should not split)
    config = Config(line_length=5, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import x", "\n", config)
    assert result == "import x"

    # Test with wrap_length different from line_length
    config = Config(line_length=80, wrap_length=20, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("from module import verylongname", "\n", config)
    assert "\\\n" in result
    assert "verylongname" in result


# LLM-generated content at query #23
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from very.long.module.name", ["import1", "import2", "import3"], config=custom_config)
    assert "import1" in result and "import2" in result and "import3" in result
    
    # Test with multi_line_output override
    result = import_statement(
        "from module", 
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    assert result.count("\n") > 0
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or result == "from module import import1, import2"
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True
    )
    result = import_statement(
        "from module", 
        ["very_long_import_name1", "very_long_import_name2", "very_long_import_name3"],
        config=balanced_config
    )
    assert isinstance(result, str)
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with remove_comments=True
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.GRID,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement(
        "from module", 
        ["import1", "import2"], 
        comments=["comment1", "comment2"],
        config=no_comments_config
    )
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #24
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom line separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n", explode=True)
    assert "\r\n" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=custom_config)
    assert "import1" in result and "import2" in result and "import3" in result
    
    # Test with multi_line_output override
    result = import_statement("from module", ["import1", "import2"], multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert "import1" in result and "import2" in result
    
    # Test with long imports that should wrap
    long_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from very_long_module_name", long_imports, config=Config(line_length=40))
    assert result.count("\n") > 0
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", [f"import{i}" for i in range(8)], config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 50 for line in lines)
    
    # Test with remove_comments
    no_comments_config = Config(ignore_comments=True)
    result = import_statement("from module", ["import1"], comments=["comment"], config=no_comments_config)
    assert "comment" not in result
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with include_trailing_comma=False
    no_comma_config = Config(include_trailing_comma=False, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement("from module", ["import1", "import2"], config=no_comma_config)
    assert result.endswith("import2\n)")
    
    # Test line_length adjustment in balanced_wrapping
    balanced_config = Config(
        line_length=30,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", ["very_long_import_name_1", "very_long_import_name_2"], 
                             config=balanced_config)
    assert result.count("\n") > 0


# LLM-generated content at query #25
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test explode mode
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result.count("\n") == 2  # Should have 2 lines in explode mode
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=45,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        ignore_comments=False,
        comment_prefix=" # ",
        use_parentheses=True
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import_name1", "very_long_import_name2"], 
                             config=custom_config)
    assert "(" in result and ")" in result
    
    # Test with line separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or "import1, import2" in result
    
    # Test balanced wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=30,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"], 
                             config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 50 for line in lines)
    
    # Test empty imports
    result = import_statement("from module", [])
    assert result == "from module"
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with trailing comma config
    comma_config = Config(
        line_length=50,
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4"], 
                             config=comma_config)
    if result.count("\n") > 0:
        assert result.strip().endswith(",") or ")" in result
    
    # Test ignore_comments
    ignore_config = Config(
        line_length=50,
        ignore_comments=True,
        comment_prefix=" # "
    )
    result = import_statement("from module", ["import1"], 
                             comments=["should be ignored"], 
                             config=ignore_config)
    assert "should be ignored" not in result


# LLM-generated content at query #26
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line wrapping with 'import' splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "from module import (" in result
    assert "very_long_name1," in result
    assert "very_long_name2," in result
    assert "\n" in result

    # Test line wrapping with 'as' splitter
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import very_long_module_name as very_long_alias", "\n", config)
    assert "very_long_module_name as" in result
    assert "\n" in result

    # Test line with comment handling
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import module1, module2  # some comment", "\n", config)
    assert "  # some comment" in result

    # Test line wrapping without parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.GRID,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=False,
        include_trailing_comma=False,
    )
    result = line("from module import name1, name2, name3, name4", "\n", config)
    assert "\\\n" in result

    # Test line with noqa comment in parentheses mode
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import module1, module2  # noqa", "\n", config)
    assert "  # noqa" in result
    assert result.endswith(")")

    # Test line that doesn't exceed length
    config = Config(line_length=100, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import short", "\n", config)
    assert result == "import short"

    # Test line with 'cimport' splitter
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from cython.module cimport func1, func2, func3", "\n", config)
    assert "cimport" in result
    assert "\n" in result

    # Test line with '.' splitter
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  #",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("module.submodule.very_long_name.another_name", "\n", config)
    assert "\n" in result


# LLM-generated content at query #27
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line with import splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert result == "from module import \\\n    very_long_name"

    # Test line with as splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import long_module_name as lmn", "\n", config)
    assert result == "import long_module_name as \\\n    lmn"

    # Test line with dot splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("module.submodule.very_long_attribute", "\n", config)
    assert result == "module.submodule.\\\n    very_long_attribute"

    # Test line with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("cimport numpy as np", "\n", config)
    assert result == "cimport numpy as \\\n    np"

    # Test line with comment handling
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ", comment_prefix="  # ")
    result = line("import module  # some comment", "\n", config)
    assert result == "import module  # some comment"

    # Test line with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import very_long_name", "\n", config)
    assert result == "from module import(\n    very_long_name,\n)"

    # Test line with parentheses and comment
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import name  # comment", "\n", config)
    assert result == "from module import(\n    name,  # comment\n)"

    # Test line with noqa comment and parentheses
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import name  # noqa", "\n", config)
    assert result == "from module import(  # noqa\n    name,\n)"

    # Test line with vertical grid grouped mode
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line("from module import very_long_name", "\n", config)
    assert result == "from module import(\n    very_long_name,\n)"

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line with custom line separator
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_name", "\r\n", config)
    assert result == "from module import \\\r\n    very_long_name"

    # Test line with wrap_length different from line_length
    config = Config(line_length=80, wrap_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert result == "from module import \\\n    very_long_name"


# LLM-generated content at query #28
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["item1", "item2"])
    assert result == "from module import item1, item2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["item1", "item2"], explode=True)
    assert result == "from module import (\n    item1,\n    item2,\n)"

    # Test with comments
    result = import_statement("from module", ["item1", "item2"], comments=["comment1", "comment2"])
    assert "# comment1" in result and "# comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        comment_prefix=" # ",
        ignore_comments=False,
        use_parentheses=True,
        balanced_wrapping=False,
    )
    result = import_statement(
        "from very_long_module_name",
        ["very_long_item_name1", "very_long_item_name2"],
        config=custom_config,
    )
    assert "from very_long_module_name import (" in result
    assert "very_long_item_name1" in result
    assert "very_long_item_name2" in result

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["item1", "item2", "item3", "item4", "item5"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        config=Config(line_length=30),
    )
    assert "import (" in result or "import \\" in result

    # Test with line_separator
    result = import_statement(
        "from module",
        ["item1", "item2"],
        line_separator="\r\n",
        config=Config(multi_line_output=Modes.VERTICAL_HANGING_INDENT, line_length=20),
    )
    assert "\r\n" in result

    # Test balanced_wrapping
    config_with_balance = Config(
        line_length=30,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    result = import_statement(
        "from module",
        ["item1", "item2", "item3", "item4", "item5"],
        config=config_with_balance,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) > 10 for line in lines)

    # Test single item import (should stay single line)
    result = import_statement("from module", ["item1"])
    assert result == "from module import item1"
    assert "\n" not in result

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with remove_comments=True
    config_no_comments = Config(ignore_comments=True)
    result = import_statement(
        "from module",
        ["item1", "item2"],
        comments=["comment1", "comment2"],
        config=config_no_comments,
    )
    assert "# comment1" not in result
    assert "# comment2" not in result

    # Test trailing comma behavior
    config_with_comma = Config(
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=30,
    )
    result = import_statement(
        "from module",
        ["item1", "item2", "item3"],
        config=config_with_comma,
    )
    assert result.endswith(",\n)")

    config_without_comma = Config(
        include_trailing_comma=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=30,
    )
    result = import_statement(
        "from module",
        ["item1", "item2", "item3"],
        config=config_without_comma,
    )
    assert not result.endswith(",\n)")


# LLM-generated content at query #29
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test wrapping with 'import' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_name" in result

    # Test wrapping with 'as' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "\\\n" in result
    assert "    vlm" in result

    # Test wrapping with '.' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from package.subpackage import name", "\n", config)
    assert "\\\n" in result

    # Test with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        indent="    ",
    )
    result = line("from module import long_name", "\n", config)
    assert "(\n" in result
    assert "    long_name,\n)" in result or "    long_name,\n)" in result

    # Test with comment handling
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import name  # some comment", "\n", config)
    assert "# some comment" in result

    # Test with noqa comment in parentheses mode
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        comment_prefix="  # ",
        indent="    ",
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "# noqa" in result
    assert result.endswith(")")

    # Test line that exactly matches line length
    config = Config(line_length=15, multi_line_output=Modes.GRID)
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from libc.math cimport sin", "\n", config)
    assert "\\\n" in result
    assert "    sin" in result

    # Test comment prefix handling
    config = Config(
        line_length=20,
        multi_line_output=Modes.GRID,
        comment_prefix="// ",
        indent="    ",
    )
    result = line("import module  # comment", "\n", config)
    assert "// comment" in result


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    config = Config(line_length=10, multi_line_output=Modes.GRID)
    result = line("import verylongmodulename", "\n", config)
    assert result == "import verylongmodulename"

    # Test line with import splitter that needs wrapping
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import verylongname", "\n", config)
    assert "\\" in result
    assert result.startswith("from module import")
    assert "verylongname" in result

    # Test line with as splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import verylongmodulename as vlm", "\n", config)
    assert "\\" in result
    assert "as" in result

    # Test line with dot splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from package.subpackage import name", "\n", config)
    assert "\\" in result

    # Test line with comment
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import module  # comment", "\n", config)
    assert "# comment" in result

    # Test NOQA mode when line is too long
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import verylongmodule", "\n", config)
    assert result == "import verylongmodule# NOQA"

    # Test NOQA mode when line already has NOQA
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        indent="    "
    )
    result = line("from module import verylongname", "\n", config)
    assert "(" in result
    assert ")" in result
    assert "," in result

    # Test with parentheses and comment containing noqa
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # "
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "# noqa" in result
    assert ")" in result

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    result = line("import import_module", "\n", config)
    assert result == "import import_module"

    # Test with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from cython cimport verylongname", "\n", config)
    assert "\\" in result
    assert "cimport" in result

    # Test line exactly at length limit
    config = Config(line_length=20, multi_line_output=Modes.GRID)
    result = line("import module123456789", "\n", config)
    assert result == "import module123456789"


# LLM-generated content at query #2
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test wrapping with import splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_name" in result

    # Test wrapping with as splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "\\\n" in result
    assert "    vlm" in result

    # Test wrapping with dot splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import package.subpackage.module", "\n", config)
    assert "\\\n" in result
    assert "    module" in result

    # Test with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        indent="    ",
    )
    result = line("from module import name1, name2", "\n", config)
    assert "(\n" in result
    assert "    name1,\n" in result
    assert "    name2,\n" in result
    assert result.endswith(")")

    # Test with comment preservation
    config = Config(
        line_length=20,
        multi_line_output=Modes.GRID,
        comment_prefix="  # ",
        indent="    ",
    )
    result = line("from module import name  # important comment", "\n", config)
    assert "# important comment" in result
    assert "\\\n" in result

    # Test with noqa comment and parentheses
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
        indent="    ",
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "# noqa" in result
    assert "(\n" in result

    # Test line starting with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    result = line("import importlib", "\n", config)
    assert result == "import importlib"

    # Test with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from cython cimport very_long_function_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_function_name" in result

    # Test wrap_length override
    config = Config(line_length=80, wrap_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from module import very_long_import_name", "\n", config)
    assert "\\\n" in result
    assert len(result.split("\n")[0]) <= 20


# LLM-generated content at query #3
#--------------------------

```python
def test_line():
    # Test basic line within length limit
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line split on 'import' with parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import (\n" in result
    assert "    very_long_name,\n)" in result

    # Test line split on 'as' with parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=False,
    )
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (\n" in result
    assert "    vlm\n)" in result

    # Test line split on '.' with parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from package.subpackage import module", "\n", config)
    assert "from package.subpackage import (\n" in result
    assert "    module,\n)" in result

    # Test line with comment preservation
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from module import name  # important comment", "\n", config)
    assert "from module import (\n" in result
    assert "    name,  # important comment\n)" in result

    # Test line with noqa comment in parentheses mode
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from module import name  # noqa", "\n", config)
    assert "from module import (  # noqa\n" in result
    assert "    name,\n)" in result

    # Test line split without parentheses (backslash continuation)
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
        indent="    ",
        comment_prefix="  # ",
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import \\\n" in result
    assert "    very_long_name" in result

    # Test line with comment and no parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=False,
    )
    result = line("from module import name  # comment", "\n", config)
    assert "from module import \\\n" in result
    assert "    name  # comment" in result

    # Test line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = line("import short", "\n", config)
    assert result == "import short"

    # Test line with trailing comma handling
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2", "\n", config)
    assert "from module import (\n" in result
    assert "    name1,\n" in result
    assert "    name2,\n)" in result

    # Test vertical grid grouped mode
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        use_parentheses=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2", "\n", config)
    assert "from module import (\n" in result
    assert "    name1,\n" in result
    assert "    name2,\n)" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config()
    config.line_length = 80
    config.multi_line_output = Modes.GRID
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "from very_long_module_name import very_long_function_name"
    result = line(long_line, "\n", config)
    assert "# NOQA" in result

    # Test line with comment that doesn't need wrapping
    config.multi_line_output = Modes.GRID
    config.line_length = 50
    result = line("import os  # comment", "\n", config)
    assert result == "import os  # comment"

    # Test line that needs wrapping with import splitter
    config.line_length = 30
    long_import = "from module.submodule import function1, function2, function3"
    result = line(long_import, "\n", config)
    assert "\\" in result or "(" in result

    # Test line with as splitter
    config.line_length = 30
    long_as = "import very_long_module_name as vlm"
    result = line(long_as, "\n", config)
    assert "\\" in result or "as" in result

    # Test line with dot splitter
    config.line_length = 30
    long_dot = "module.submodule.very_long_function_name"
    result = line(long_dot, "\n", config)
    assert "\\" in result or "." in result

    # Test line with cimport splitter
    config.line_length = 30
    long_cimport = "from cython_module cimport func1, func2, func3"
    result = line(long_cimport, "\n", config)
    assert "\\" in result or "cimport" in result

    # Test with parentheses mode
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.line_length = 40
    long_parentheses = "from module import func1, func2, func3, func4"
    result = line(long_parentheses, "\n", config)
    assert "(" in result and ")" in result

    # Test with vertical hanging indent mode
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    result = line(long_parentheses, "\n", config)
    assert "\n" in result

    # Test with vertical grid grouped mode
    config.multi_line_output = Modes.VERTICAL_GRID_GROUPED
    result = line(long_parentheses, "\n", config)
    assert "\n" in result

    # Test line with noqa comment already present
    config.multi_line_output = Modes.NOQA
    line_with_noqa = "import os  # NOQA"
    result = line(line_with_noqa, "\n", config)
    assert result == line_with_noqa

    # Test line with comment containing noqa in parentheses mode
    config.use_parentheses = True
    config.multi_line_output = Modes.GRID
    line_with_noqa_comment = "from module import func1, func2  # noqa"
    result = line(line_with_noqa_comment, "\n", config)
    assert "noqa" in result

    # Test line exactly at length limit
    config.line_length = 20
    exact_length = "import os.path.join"
    result = line(exact_length, "\n", config)
    assert result == exact_length

    # Test with custom indent
    config.indent = "    "
    config.line_length = 30
    long_line = "from module import func1, func2"
    result = line(long_line, "\n", config)
    if "\\" in result:
        assert result.startswith("from module import")

    # Test with wrap_length different from line_length
    config.wrap_length = 25
    config.line_length = 80
    long_line = "from module import func1, func2, func3"
    result = line(long_line, "\n", config)
    assert len(result.split("\n")[0]) <= 25

    # Test empty line
    result = line("", "\n", config)
    assert result == ""

    # Test line with only comment
    result = line("# comment only", "\n", config)
    assert result == "# comment only"


# LLM-generated content at query #5
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    long_line = "a" * 100
    result = line(long_line, "\n", config)
    assert result == long_line

    # Test line with import splitter
    content = "from very_long_module_name import very_long_function_name"
    config = Config(line_length=40, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with as splitter
    content = "import very_long_module_name as very_long_alias_name"
    config = Config(line_length=40, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with dot splitter
    content = "module.submodule.very_long_attribute_name"
    config = Config(line_length=30, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with comment
    content = "import os  # comment"
    config = Config(line_length=20, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "# comment" in result

    # Test NOQA mode with long line
    content = "a" * 100
    config = Config(line_length=80, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert "NOQA" in result

    # Test NOQA mode when already has NOQA
    content = "a" * 100 + "  # NOQA"
    config = Config(line_length=80, multi_line_output=Modes.NOQA)
    result = line(content, "\n", config)
    assert result == content

    # Test with parentheses and trailing comma
    content = "from module import very_long_name"
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line(content, "\n", config)
    assert "(" in result
    assert ")" in result
    assert "," in result

    # Test with comment containing noqa
    content = "from module import name  # noqa"
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line(content, "\n", config)
    assert "# noqa" in result

    # Test with comment prefix
    content = "import os  # comment"
    config = Config(
        line_length=20,
        multi_line_output=Modes.GRID,
        comment_prefix=" # ",
        use_parentheses=True,
    )
    result = line(content, "\n", config)
    assert "# comment" in result

    # Test cimport splitter
    content = "from cython_module cimport very_long_function"
    config = Config(line_length=40, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line that starts with splitter (should not split)
    content = "import os"
    config = Config(line_length=5, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert result == content

    # Test with wrap_length different from line_length
    content = "from module import very_long_name"
    config = Config(
        line_length=100,
        wrap_length=30,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
    )
    result = line(content, "\n", config)
    assert len(result.split("\n")) > 1


# LLM-generated content at query #6
#--------------------------

```python
def test_line():
    # Test basic line without wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    assert line("import os", "\n", config) == "import os"

    # Test line exceeding length with NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "import " + "very_long_module_name_" * 5
    result = line(long_line, "\n", config)
    assert result.endswith("# NOQA")
    assert "# NOQA" in result

    # Test line with comment
    config.multi_line_output = Modes.GRID
    config.comment_prefix = "  # "
    line_with_comment = "import os  # comment"
    assert line(line_with_comment, "\n", config) == line_with_comment

    # Test wrapping with 'import' splitter
    config.line_length = 20
    config.wrap_length = None
    config.indent = "    "
    content = "from module import very_long_name"
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "    very_long_name" in result

    # Test wrapping with parentheses
    config.use_parentheses = True
    config.include_trailing_comma = True
    content = "from module import very_long_name"
    result = line(content, "\n", config)
    assert "(" in result
    assert ")" in result
    assert ",\n" in result or result.endswith(",")

    # Test with 'as' splitter
    content = "import very_long_module_name as vlm"
    result = line(content, "\n", config)
    assert "as" in result
    assert "\\\n" not in result

    # Test with comment and noqa
    config.ignore_comments = False
    content = "import os  # noqa"
    result = line(content, "\n", config)
    assert "noqa" in result

    # Test vertical hanging indent mode
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    content = "from module import name1, name2, name3"
    result = line(content, "\n", config)
    assert line_separator in result
    assert ")" in result

    # Test line within length limit
    config.line_length = 100
    short_line = "import os"
    assert line(short_line, "\n", config) == short_line

    # Test with custom line separator
    result = line("from module import name", "\r\n", config)
    assert "\r\n" in result or result == "from module import name"

    # Test trailing comma handling
    config.include_trailing_comma = True
    content = "from module import very_long_import_name"
    result = line(content, "\n", config)
    if "(" in result:
        assert result.endswith(",)") or ",\n)" in result

    # Test no wrapping when wrap_mode is NOQA but NOQA already present
    config.multi_line_output = Modes.NOQA
    content = "import os  # NOQA"
    assert line(content, "\n", config) == content


# LLM-generated content at query #7
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert result == "from module import import1, import2  # comment1  # comment2"
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        ignore_comments=False,
        use_parentheses=True,
        balanced_wrapping=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from very_long_module_name", 
                            ["very_long_import_name1", "very_long_import_name2"],
                            config=custom_config)
    assert "from very_long_module_name import (" in result
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                            ["import1", "import2", "import3", "import4", "import5"],
                            multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert result.count("\n") > 0
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator
    
    # Test balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=30,
        indent="    ",
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", 
                            ["very_long_import_name_" + str(i) for i in range(10)],
                            config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) - 1
    
    # Test empty imports
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import with explode
    result = import_statement("from module", ["single_import"], explode=True)
    assert result == "from module import (\n    single_import,\n)"
    
    # Test with remove_comments=True
    no_comments_config = Config(
        line_length=50,
        ignore_comments=True,
        comment_prefix="  # "
    )
    result = import_statement("from module", ["import1", "import2"], 
                            comments=["comment1", "comment2"],
                            config=no_comments_config)
    assert "#" not in result
    
    # Test trailing comma behavior
    trailing_comma_config = Config(
        line_length=20,
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", 
                            ["import1", "import2", "import3"],
                            config=trailing_comma_config)
    assert result.endswith(",\n)")


# LLM-generated content at query #8
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line that exceeds length but has NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line wrapping with 'import' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from package import very_long_module_name", "\n", config)
    assert "\\\n" in result
    assert "    very_long_module_name" in result

    # Test line wrapping with 'as' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("import very_long_module_name as vln", "\n", config)
    assert "\\\n" in result
    assert "    vln" in result

    # Test line wrapping with '.' splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line("from package.subpackage import module", "\n", config)
    assert "\\\n" in result
    assert "    module" in result

    # Test line with comment preservation
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ", comment_prefix="  # ")
    result = line("from package import module  # important comment", "\n", config)
    assert "important comment" in result

    # Test line wrapping with parentheses
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT, 
                   indent="    ", use_parentheses=True, include_trailing_comma=True)
    result = line("from package import very_long_module_name", "\n", config)
    assert "(\n" in result
    assert "    very_long_module_name,\n)" in result or "    very_long_module_name,\n)" in result

    # Test line wrapping with vertical grid grouped mode
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_GRID_GROUPED, 
                   indent="    ", use_parentheses=True, include_trailing_comma=False)
    result = line("from package import very_long_module_name", "\n", config)
    assert "(\n" in result
    assert ")" in result

    # Test line with noqa comment and parentheses
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                   indent="    ", use_parentheses=True, comment_prefix="  # ")
    result = line("from package import module  # noqa", "\n", config)
    assert "  # noqa" in result
    assert ")" in result

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, indent="    ")
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line with trailing comma handling
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                   indent="    ", use_parentheses=True, include_trailing_comma=True)
    result = line("from package import mod1, mod2  # comment", "\n", config)
    assert "mod2,  # comment" in result or "mod2,  # comment" in result

    # Test line without trailing comma when include_trailing_comma is False
    config = Config(line_length=20, multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                   indent="    ", use_parentheses=True, include_trailing_comma=False)
    result = line("from package import mod1, mod2  # comment", "\n", config)
    assert "mod2  # comment" in result and "mod2," not in result


# LLM-generated content at query #9
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1"], comments=["comment1"])
    assert result == "from module import import1  # comment1"

    # Test with multiple comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "import1  # comment1" in result
    assert "import2  # comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        ignore_comments=False,
        use_parentheses=True,
    )
    
    # Test long import that should wrap
    long_imports = ["very_long_import_name_" + str(i) for i in range(5)]
    result = import_statement("from very_long_module_name", long_imports, config=custom_config)
    assert "(" in result  # Should use parentheses
    assert "\n" in result  # Should be multi-line

    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=80,
        wrap_length=60,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=True,
        ignore_comments=False,
        use_parentheses=True,
    )
    
    result = import_statement("from module", ["import1", "import2", "import3", "import4"], config=balanced_config)
    # Should produce balanced output
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) > 10 for line in lines[:-1])

    # Test with custom line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator
    result = import_statement("from module", ["import1", "import2"], explode=True, line_separator="\r\n")
    assert "\r\n" in result  # Multi-line should use custom separator

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    # Should use vertical grid grouped formatting
    assert result.count("\n") >= 2

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test with ignore_comments
    no_comment_config = Config(
        line_length=80,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
    )
    result = import_statement("from module", ["import1"], comments=["comment1"], config=no_comment_config)
    assert "#" not in result  # Comments should be ignored

    # Test trailing comma behavior
    trailing_comma_config = Config(
        line_length=30,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=trailing_comma_config)
    assert result.endswith(",\n)")  # Should have trailing comma

    no_trailing_comma_config = Config(
        line_length=30,
        indent="    ",
        include_trailing_comma=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=no_trailing_comma_config)
    assert not result.endswith(",\n)")  # Should not have trailing comma


# LLM-generated content at query #10
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom line separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n", explode=True)
    assert "\r\n" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=custom_config)
    assert "import1," in result and "import2," in result

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    assert len(result.split("\n")) > 1

    # Test with long imports that require wrapping
    long_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from module", long_imports, config=DEFAULT_CONFIG)
    assert isinstance(result, str)

    # Test with empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test balanced_wrapping behavior
    balanced_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement(
        "from module",
        [f"import{i}" for i in range(5)],
        config=balanced_config,
    )
    assert isinstance(result, str)

    # Test with remove_comments=True
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from module",
        ["import1", "import2"],
        comments=["comment1", "comment2"],
        config=no_comments_config,
    )
    assert "comment1" not in result and "comment2" not in result

    # Test line wrapping for single line result
    short_config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from verylongmodulename",
        ["import1", "import2"],
        config=short_config,
    )
    assert isinstance(result, str)


# LLM-generated content at query #11
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #", indent="    ")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length with NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "from very_long_module_name import very_long_function_name"
    result = line(long_line, "\n", config)
    assert result == f"{long_line}{config.comment_prefix} NOQA"

    # Test line with comment and NOQA already present
    line_with_noqa = "import os  # NOQA"
    result = line(line_with_noqa, "\n", config)
    assert result == line_with_noqa

    # Test line with import splitter
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 50
    config.use_parentheses = True
    config.include_trailing_comma = True
    long_import = "from module.submodule.anothersubmodule import some_function"
    result = line(long_import, "\n", config)
    assert "\\" not in result
    assert result.startswith("from module.submodule.anothersubmodule import (")
    assert "some_function" in result

    # Test line with as splitter
    config.line_length = 30
    long_as = "import very_long_module_name as vlm"
    result = line(long_as, "\n", config)
    assert "as" in result
    assert "\\" in result or "(" in result

    # Test line with dot splitter
    config.line_length = 40
    long_dot = "module.submodule.anothersubmodule.function"
    result = line(long_dot, "\n", config)
    assert "\\" in result or "(" in result

    # Test line with comment handling
    config.line_length = 50
    line_with_comment = "from module import something  # some comment"
    result = line(line_with_comment, "\n", config)
    assert "# some comment" in result

    # Test line with noqa comment and parentheses
    config.line_length = 40
    line_with_noqa_comment = "from module import something  # noqa"
    result = line(line_with_noqa_comment, "\n", config)
    assert "# noqa" in result
    assert result.endswith(")")

    # Test line without parentheses
    config.use_parentheses = False
    config.line_length = 30
    long_line_no_parens = "from module import something"
    result = line(long_line_no_parens, "\n", config)
    assert "\\" in result

    # Test line with trailing comma handling
    config.use_parentheses = True
    config.include_trailing_comma = True
    config.line_length = 35
    line_for_comma = "from module import item1, item2"
    result = line(line_for_comma, "\n", config)
    assert "," in result
    assert result.count(",") > 1

    # Test line with cimport splitter
    config.line_length = 30
    cimport_line = "from module cimport something"
    result = line(cimport_line, "\n", config)
    assert "cimport" in result
    assert "\\" in result or "(" in result

    # Test line that starts with splitter (should not split)
    config.line_length = 10
    starts_with_import = "import module"
    result = line(starts_with_import, "\n", config)
    assert result == starts_with_import

    # Test line with comment prefix in last line
    config.line_length = 40
    config.comment_prefix = "  # "
    complex_line = "from module import (item1, item2)  # comment"
    result = line(complex_line, "\n", config)
    assert result.endswith(")  # comment")

    # Test line with wrap_length different from line_length
    config.wrap_length = 30
    config.line_length = 80
    wrap_line = "from very.long.module.path import function"
    result = line(wrap_line, "\n", config)
    assert len(result.split("\n")[0]) <= 30

    # Test line with no wrapping when under length
    config.line_length = 100
    short_line = "import os"
    result = line(short_line, "\n", config)
    assert result == short_line


# LLM-generated content at query #12
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result
    assert "comment2" in result
    
    # Test explode mode
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result.count("\n") == 2  # Should have 2 line breaks for 2 imports
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2", "very_long_import3"],
                             config=custom_config)
    assert "    " in result  # Should have indentation
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result or result.count("\r\n") > 0
    
    # Test balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1]) - 1
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4"],
                             multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert "import" in result
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module"
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with trailing comma config
    comma_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             config=comma_config)
    if result.count("\n") > 0:
        assert result.strip().endswith(",") or ")" in result


# LLM-generated content at query #13
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom line separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line should not have separator
    
    # Test with multi_line_output mode
    from .wrap_modes import WrapModes as Modes
    result = import_statement("from module", ["import1"] * 10, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    assert result.count("\n") > 0
    
    # Test with balanced_wrapping in config
    config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=True,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        ignore_comments=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1"] * 20, config=config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 80 for line in lines)
    
    # Test with long imports that should wrap
    long_imports = ["very_long_import_name_" + str(i) for i in range(10)]
    result = import_statement("from module", long_imports)
    assert isinstance(result, str)
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module"
    
    # Test with trailing comma config
    config_no_comma = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=False,
        ignore_comments=False,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=config_no_comma)
    if result.count("\n") > 0:
        assert not result.strip().endswith(",")
    
    # Test with ignore_comments config
    config_ignore_comments = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        ignore_comments=True,
        use_parentheses=True
    )
    result = import_statement("from module", ["import1"], comments=["comment"], config=config_ignore_comments)
    assert "comment" not in result


# LLM-generated content at query #14
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2", "import3"], config=custom_config)
    assert "import1," in result and "import2," in result and "import3," in result
    
    # Test with multi_line_output override
    result = import_statement(
        "from module", 
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        config=DEFAULT_CONFIG
    )
    assert result.count("\n") > 0
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or "import1, import2" in result
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement(
        "from module", 
        ["very_long_import_name1", "very_long_import_name2", "very_long_import_name3"],
        config=balanced_config
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 50 for line in lines)
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test empty imports
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test with remove_comments
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.GRID,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from module", 
        ["import1", "import2"], 
        comments=["comment1", "comment2"],
        config=no_comments_config
    )
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #15
#--------------------------

```python
def test_line():
    from .settings import Config
    from .wrap_modes import WrapModes as Modes

    # Test 1: Simple line within length limit
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test 2: Line exceeding length with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test 3: Line with comment and NOQA already present
    config = Config(line_length=10, multi_line_output=Modes.NOQA)
    result = line("import mod  # NOQA", "\n", config)
    assert result == "import mod  # NOQA"

    # Test 4: Line split on "import " with parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "(\n" in result and "very_long_name1" in result and "very_long_name2" in result

    # Test 5: Line split on "as " with parentheses
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "import very_long_module_name as very_long_alias_name", "\n", config
    )
    assert "as" in result and "very_long_alias_name" in result

    # Test 6: Line with comment preservation
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=False,
        comment_prefix="  # ",
    )
    result = line(
        "from module import name1, name2  # important comment", "\n", config
    )
    assert "# important comment" in result

    # Test 7: Line split on "." with parentheses
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "module.submodule.very_long_attribute_name", "\n", config
    )
    assert "(\n" in result and "very_long_attribute_name" in result

    # Test 8: Line with noqa comment and parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "from module import name1, name2  # noqa", "\n", config
    )
    assert "# noqa" in result and result.endswith(")")

    # Test 9: Line without parentheses (backslash continuation)
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
        indent="    ",
        comment_prefix="  # ",
    )
    result = line(
        "from module import very_long_name1, very_long_name2", "\n", config
    )
    assert "\\\n" in result and "very_long_name1" in result

    # Test 10: Empty line
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("", "\n", config)
    assert result == ""

    # Test 11: Line exactly at length limit
    config = Config(line_length=10, multi_line_output=Modes.GRID)
    result = line("0123456789", "\n", config)
    assert result == "0123456789"

    # Test 12: Line with trailing comma handling
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "from module import name1, name2, name3", "\n", config
    )
    assert "," in result.split("\n")[-1] or result.endswith(",")

    # Test 13: Vertical grid grouped mode
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        use_parentheses=True,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "from module import name1, name2, name3, name4", "\n", config
    )
    assert "(\n" in result and ")" in result

    # Test 14: Comment with noqa not in parentheses mode
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        use_parentheses=False,
        indent="    ",
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line(
        "import module  # noqa comment here", "\n", config
    )
    assert "# noqa comment here" in result


# LLM-generated content at query #16
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with NOQA mode
    config.multi_line_output = Modes.NOQA
    long_line = "from very_long_module_name import very_long_function_name_that_exceeds_line_length"
    result = line(long_line, "\n", config)
    assert "NOQA" in result

    # Test line with comment and import splitter
    config.multi_line_output = Modes.VERTICAL_HANGING_INDENT
    config.line_length = 50
    config.use_parentheses = True
    config.indent = "    "
    content = "from module import something, another_thing  # some comment"
    result = line(content, "\n", config)
    assert "some comment" in result
    assert "(" in result
    assert "\n" in result

    # Test line with as splitter
    config.line_length = 30
    content = "import very_long_module_name as vlm"
    result = line(content, "\n", config)
    assert "as" in result
    assert "\n" in result

    # Test line with dot splitter
    config.line_length = 40
    content = "from package.subpackage.module import function"
    result = line(content, "\n", config)
    assert "." in result
    assert "\n" in result

    # Test line with trailing comma handling
    config.include_trailing_comma = True
    config.line_length = 50
    content = "from module import item1, item2, item3, item4, item5  # noqa"
    result = line(content, "\n", config)
    assert "," in result
    assert "noqa" in result

    # Test line without parentheses
    config.use_parentheses = False
    config.line_length = 40
    content = "import very_very_long_module_name_here"
    result = line(content, "\n", config)
    assert "\\" in result
    assert "\n" in result

    # Test line with comment prefix handling
    config.comment_prefix = "# "
    config.line_length = 60
    content = "from module import something  # inline comment here"
    result = line(content, "\n", config)
    assert "# inline comment here" in result

    # Test line with cimport splitter
    config.line_length = 30
    content = "from cython_module cimport function"
    result = line(content, "\n", config)
    assert "cimport" in result
    assert "\n" in result

    # Test line that exactly matches line length
    config.line_length = 20
    content = "import os" + " " * 12  # Total length 20
    result = line(content, "\n", config)
    assert result == content

    # Test line with vertical grid grouped mode
    config.multi_line_output = Modes.VERTICAL_GRID_GROUPED
    config.line_length = 50
    config.use_parentheses = True
    content = "from module import item1, item2, item3, item4, item5, item6"
    result = line(content, "\n", config)
    assert "VERTICAL_GRID_GROUPED" not in result  # Should not contain mode name
    assert "\n" in result


# LLM-generated content at query #17
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has no splitter
    long_line = "import " + "very_long_module_name" * 10
    result = line(long_line, "\n", config)
    assert result == long_line

    # Test line with import splitter
    content = "from module import very_long_import_name_that_exceeds_line_length"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert result.startswith("from module import")
    assert "    very_long_import_name_that_exceeds_line_length" in result

    # Test line with dot splitter
    content = "module.very_long_attribute_name_that_exceeds_line_length"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "module." in result
    assert "    very_long_attribute_name_that_exceeds_line_length" in result

    # Test line with as splitter
    content = "module as very_long_alias_name_that_exceeds_line_length"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert "\\\n" in result
    assert "module as" in result
    assert "    very_long_alias_name_that_exceeds_line_length" in result

    # Test line with comment
    content = "import os  # some comment"
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ", comment_prefix="  # ")
    result = line(content, "\n", config)
    assert "  # some comment" in result

    # Test NOQA mode with long line
    content = "import " + "x" * 100
    config = Config(line_length=50, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, "\n", config)
    assert "  # NOQA" in result

    # Test NOQA mode when already has NOQA comment
    content = "import os  # NOQA"
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line(content, "\n", config)
    assert result == content

    # Test with parentheses and trailing comma
    content = "from module import very_long_name"
    config = Config(
        line_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line(content, "\n", config)
    assert "(" in result
    assert ")" in result
    assert ",\n" in result or result.endswith(",")

    # Test with comment and noqa in parentheses mode
    content = "from module import name  # noqa"
    config = Config(
        line_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # "
    )
    result = line(content, "\n", config)
    assert "  # noqa" in result
    assert ")" in result

    # Test line that starts with splitter (should not split)
    content = "import os"
    config = Config(line_length=5, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert result == content

    # Test with cimport splitter
    content = "from module cimport very_long_name"
    config = Config(line_length=30, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert "cimport" in result
    assert "\\\n" in result

    # Test multiple splits
    content = "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p.q.r.s.t.u.v.w.x.y.z"
    config = Config(line_length=20, multi_line_output=Modes.GRID, indent="    ")
    result = line(content, "\n", config)
    assert result.count("\\\n") >= 2

    # Test empty line
    result = line("", "\n", config)
    assert result == ""

    # Test line exactly at length limit
    content = "x" * 80
    config = Config(line_length=80, multi_line_output=Modes.GRID)
    result = line(content, "\n", config)
    assert result == content


# LLM-generated content at query #18
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert result == "from module import import1, import2  # comment1  # comment2"

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", ["import1", "import2", "import3"], config=custom_config)
    assert "from very_long_module_name import (" in result
    assert "import1," in result
    assert "import2," in result
    assert "import3," in result

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    assert result.count("\n") > 0

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't use separator

    # Test with many imports that force wrapping
    many_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from module", many_imports)
    assert len(result) <= DEFAULT_CONFIG.line_length or "\n" in result

    # Test balanced_wrapping
    balanced_config = Config(
        line_length=80,
        wrap_length=60,
        indent="    ",
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    result = import_statement(
        "from module",
        [f"very_long_import_name_{i}" for i in range(5)],
        config=balanced_config,
    )
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 80 for line in lines)

    # Test with remove_comments
    no_comments_config = Config(ignore_comments=True)
    result = import_statement(
        "from module",
        ["import1", "import2"],
        comments=["comment1", "comment2"],
        config=no_comments_config,
    )
    assert "#" not in result

    # Test empty from_imports
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["import1"])
    assert result == "from module import import1"


# LLM-generated content at query #19
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "# comment1" in result and "# comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very.long.module.name", ["import1", "import2", "import3"], config=custom_config)
    assert "from very.long.module.name import (" in result
    assert "import1," in result
    assert "import2," in result
    assert "import3," in result

    # Test with many imports causing wrapping
    many_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from module", many_imports, config=custom_config)
    assert result.count("\n") > 1

    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement("from module", many_imports, config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert len(lines[-1]) >= min(len(line) for line in lines[:-1])

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or result == "from module import import1, import2"

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    assert "import (" in result

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test with remove_comments=True
    no_comments_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from module",
        ["import1", "import2"],
        comments=["comment1", "comment2"],
        config=no_comments_config,
    )
    assert "# comment1" not in result
    assert "# comment2" not in result


# LLM-generated content at query #20
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line that exceeds length but has NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with NOQA mode but already has NOQA comment
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line that needs wrapping with 'import' splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import (\n" in result
    assert "    very_long_name," in result

    # Test line that needs wrapping with 'as' splitter
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as vlm" == result

    # Test line with comment that doesn't contain noqa
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2  # some comment", "\n", config)
    assert "  # some comment" in result

    # Test line with comment containing noqa
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2  # noqa", "\n", config)
    assert "  # noqa" in result

    # Test line with '.' splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("module.submodule.verylongname", "\n", config)
    assert "module.submodule.verylongname" == result

    # Test line without parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=False,
        include_trailing_comma=False,
    )
    result = line("from module import very_long_name", "\n", config)
    assert "from module import \\\n" in result
    assert "    very_long_name" in result

    # Test line starting with splitter (should not split)
    config = Config(
        line_length=10,
        wrap_length=10,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test vertical grid grouped mode
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2, name3", "\n", config)
    assert "from module import (\n" in result

    # Test with trailing comma when include_trailing_comma is True
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=True,
    )
    result = line("from module import name1, name2", "\n", config)
    assert result.endswith(",") or "name2," in result

    # Test without trailing comma when include_trailing_comma is False
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        comment_prefix="  # ",
        use_parentheses=True,
        include_trailing_comma=False,
    )
    result = line("from module import name1, name2", "\n", config)
    assert not result.endswith(",") and "name2," not in result


# LLM-generated content at query #21
#--------------------------

```python
def test_line():
    # Test basic line with no wrapping needed
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line exceeding length with import splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("from very_long_module_name import very_long_function_name", "\n", config)
    assert "\\" in result
    assert result.startswith("from very_long_module_name import")
    assert "very_long_function_name" in result

    # Test line with comment
    config = Config(line_length=30, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import os  # comment", "\n", config)
    assert result == "import os  # comment"

    # Test line with NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import verylongmodulename", "\n", config)
    assert result == "import verylongmodulename  # NOQA"

    # Test line with NOQA mode and existing NOQA comment
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  #")
    result = line("import mod  # NOQA", "\n", config)
    assert result == "import mod  # NOQA"

    # Test line with parentheses and trailing comma
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  #",
    )
    result = line("from module import function", "\n", config)
    assert result.startswith("from module import (")
    assert "function," in result
    assert result.endswith(")")

    # Test line with comment and noqa in parentheses mode
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  #",
    )
    result = line("from module import function  # noqa", "\n", config)
    assert "(  # noqa" in result
    assert result.endswith(")")

    # Test line with as splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import longmodule as lm", "\n", config)
    assert "\\" in result
    assert "import longmodule as" in result
    assert "lm" in result

    # Test line with cimport splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("cimport numpy as np", "\n", config)
    assert "\\" in result
    assert "cimport numpy as" in result
    assert "np" in result

    # Test line with dot splitter
    config = Config(line_length=20, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("module.submodule.function", "\n", config)
    assert "\\" in result
    assert "module.submodule" in result
    assert "function" in result

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, comment_prefix="  #")
    result = line("import module", "\n", config)
    assert result == "import module"

    # Test line with comment and trailing comma logic
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  #",
    )
    result = line("from m import a, b  # comment", "\n", config)
    assert "a," in result
    assert "b  # comment" in result
    assert result.endswith(")")

    # Test line with vertical grid grouped mode
    config = Config(
        line_length=20,
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  #",
    )
    result = line("from module import function", "\n", config)
    assert result.startswith("from module import (")
    assert result.endswith(")")


# LLM-generated content at query #22
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        comment_prefix="  # ",
        include_trailing_comma=True,
        ignore_comments=False,
        use_parentheses=True,
        balanced_wrapping=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2", "very_long_import3"],
                             config=custom_config)
    assert "(" in result and ")" in result
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert result.count("\n") > 0
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or "import1, import2" in result
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=30,
        wrap_length=30,
        indent="    ",
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        assert all(len(line) <= 30 for line in lines)
    
    # Test with ignore_comments
    ignore_config = Config(
        ignore_comments=True,
        comment_prefix="  # "
    )
    result = import_statement("from module", ["import1"], 
                             comments=["This comment should be ignored"],
                             config=ignore_config)
    assert "This comment should be ignored" not in result
    
    # Test trailing comma behavior
    trailing_comma_config = Config(
        line_length=20,
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3"],
                             config=trailing_comma_config)
    if "(" in result and ")" in result:
        assert result.strip().endswith(",)")


# LLM-generated content at query #23
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with explode=True creates multi-line
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        comment_prefix="  # ",
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", 
                             ["very_long_import1", "very_long_import2", "very_long_import3"],
                             config=custom_config)
    assert "very_long_import1" in result
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=30,
        wrap_length=30,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=True,
        comment_prefix="  # ",
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4", "import5"],
                             config=balanced_config)
    lines = result.split("\n")
    assert len(lines) > 1
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result or result == "from module import import1, import2"
    
    # Test with multi_line_output override
    result = import_statement("from module", 
                             ["import1", "import2", "import3", "import4"],
                             multi_line_output=Modes.VERTICAL_HANGING_INDENT,
                             config=DEFAULT_CONFIG)
    assert "import1" in result
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"
    
    # Test with remove_comments
    no_comments_config = Config(
        line_length=80,
        wrap_length=80,
        indent="    ",
        include_trailing_comma=False,
        multi_line_output=Modes.GRID,
        balanced_wrapping=False,
        comment_prefix="  # ",
        ignore_comments=True,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2"], 
                             comments=["comment1", "comment2"],
                             config=no_comments_config)
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #24
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["function1", "function2"])
    assert result == "from module import function1, function2"
    
    # Test with comments
    result = import_statement("from module", ["func1", "func2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result
    
    # Test explode mode
    result = import_statement("from module", ["func1", "func2"], explode=True)
    assert "\n" in result  # Should be multi-line in explode mode
    
    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        comment_prefix=" # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True
    )
    result = import_statement("from long_module_name", ["very_long_function_name_1", "very_long_function_name_2"], config=custom_config)
    assert len(result.split("\n")) > 1  # Should wrap to multiple lines
    
    # Test with line_separator
    result = import_statement("from module", ["func1", "func2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator
    
    # Test balanced_wrapping
    balanced_config = Config(
        line_length=50,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT
    )
    long_imports = [f"function{i}" for i in range(10)]
    result = import_statement("from module", long_imports, config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        # Check that balanced wrapping attempted to balance line lengths
        assert all(len(line) > 10 for line in lines[:-1])
    
    # Test with trailing comma
    trailing_comma_config = Config(
        line_length=30,
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        use_parentheses=True
    )
    result = import_statement("from module", ["func1", "func2", "func3"], config=trailing_comma_config)
    assert result.strip().endswith(",") or ",\n)" in result
    
    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "
    
    # Test single import
    result = import_statement("from module", ["single_func"])
    assert result == "from module import single_func"
    
    # Test with multi_line_output override
    result = import_statement("from module", ["func1", "func2", "func3", "func4"], 
                            multi_line_output=Modes.VERTICAL_GRID_GROUPED)
    assert "import (" in result or "\n" in result
    
    # Test comment removal when ignore_comments=True
    no_comments_config = Config(ignore_comments=True)
    result = import_statement("from module", ["func1"], comments=["Should be removed"], config=no_comments_config)
    assert "Should be removed" not in result


# LLM-generated content at query #25
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result and "comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from very_long_module_name", ["import1", "import2", "import3"], config=custom_config)
    assert "(\n" in result and ",\n" in result

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Should be single line

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    assert "(\n" in result

    # Test with balanced_wrapping
    config_with_balance = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    result = import_statement(
        "from module",
        ["very_long_import_name_1", "very_long_import_name_2", "short"],
        config=config_with_balance,
    )
    assert isinstance(result, str)

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test with remove_comments
    config_no_comments = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.GRID,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=True,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from module", ["import1", "import2"], comments=["comment1", "comment2"], config=config_no_comments
    )
    assert "comment1" not in result and "comment2" not in result


# LLM-generated content at query #26
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result
    assert "comment2" in result

    # Test with explode=True
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert "\n" in result
    assert "import1" in result
    assert "import2" in result

    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="# ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    result = import_statement("from module", ["import1", "import2"], config=custom_config)
    assert "import1" in result
    assert "import2" in result

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" not in result  # Single line shouldn't have separator

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4", "import5"],
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        config=Config(line_length=20),
    )
    assert "\n" in result
    assert "import1" in result

    # Test with many imports that should wrap
    result = import_statement(
        "from very_long_module_name",
        ["import1", "import2", "import3", "import4", "import5", "import6", "import7"],
        config=Config(line_length=50),
    )
    assert "import1" in result

    # Test with balanced_wrapping
    config_with_balance = Config(
        line_length=80,
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
    )
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4", "import5"],
        config=config_with_balance,
    )
    assert "import1" in result

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module"

    # Test single import
    result = import_statement("from module", ["single_import"])
    assert result == "from module import single_import"

    # Test with trailing comma config
    config_with_comma = Config(
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=30,
    )
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4"],
        config=config_with_comma,
    )
    assert "," in result


# LLM-generated content at query #27
#--------------------------

```python
def test_line():
    # Test basic line that doesn't need wrapping
    config = Config(line_length=80, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import os", "\n", config)
    assert result == "import os"

    # Test line that exceeds length but has NOQA mode
    config = Config(line_length=10, multi_line_output=Modes.NOQA, comment_prefix="  # ")
    result = line("import very_long_module_name", "\n", config)
    assert result == "import very_long_module_name  # NOQA"

    # Test line with NOQA mode but already has NOQA comment
    result = line("import module  # NOQA", "\n", config)
    assert result == "import module  # NOQA"

    # Test line that needs wrapping with 'import' splitter
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line("from module import very_long_name1, very_long_name2", "\n", config)
    assert "from module import (" in result
    assert "very_long_name1," in result
    assert "very_long_name2," in result
    assert result.count("\n") >= 1

    # Test line that needs wrapping with 'as' splitter
    result = line("import very_long_module_name as vlm", "\n", config)
    assert "import very_long_module_name as (" in result or "import very_long_module_name as \\" in result

    # Test line that needs wrapping with '.' splitter
    result = line("from package.subpackage import name", "\n", config)
    assert "from package.subpackage import (" in result or "from package.subpackage import \\" in result

    # Test line with comment that doesn't contain 'noqa'
    config = Config(
        line_length=30,
        wrap_length=30,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line("import mod1, mod2, mod3, mod4  # some comment", "\n", config)
    assert "  # some comment" in result
    assert result.endswith(")")

    # Test line with 'noqa' in comment
    result = line("import mod1, mod2, mod3, mod4  # noqa", "\n", config)
    assert "  # noqa" in result
    assert result.endswith(")  # noqa") or result.endswith("  # noqa")

    # Test line without parentheses
    config = Config(
        line_length=20,
        wrap_length=20,
        multi_line_output=Modes.GRID,
        indent="    ",
        use_parentheses=False,
        comment_prefix="  # ",
    )
    result = line("import very_long_name1, very_long_name2", "\n", config)
    assert "\\\n" in result
    assert "    " in result  # Contains indent

    # Test line with trailing comma handling
    config = Config(
        line_length=25,
        wrap_length=25,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        indent="    ",
        use_parentheses=True,
        include_trailing_comma=True,
        comment_prefix="  # ",
    )
    result = line("import name1, name2, name3", "\n", config)
    assert result.endswith(",") or "name3," in result

    # Test line with cimport splitter
    result = line("from libc.stdio cimport printf, scanf", "\n", config)
    assert "cimport" in result
    assert "printf" in result

    # Test line that starts with splitter (should not split)
    config = Config(line_length=10, multi_line_output=Modes.GRID, comment_prefix="  # ")
    result = line("import os", "\n", config)
    assert result == "import os"


# LLM-generated content at query #28
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"
    
    # Test with comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "comment1" in result
    assert "comment2" in result
    
    # Test with explode=True
    result = import_statement("from module", ["import1", "import2", "import3"], explode=True)
    lines = result.split("\n")
    assert len(lines) > 1
    assert lines[0] == "from module import ("
    assert lines[-1] == ")"
    
    # Test with custom config
    custom_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    
    # Test long import that should wrap
    long_imports = [f"import{i}" for i in range(10)]
    result = import_statement("from very_long_module_name", long_imports, config=custom_config)
    assert result.count("\n") > 0
    
    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n")
    assert "\r\n" in result
    
    # Test with balanced_wrapping
    balanced_config = Config(
        line_length=50,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=True,
        use_parentheses=True,
    )
    
    balanced_imports = [f"very_long_import_name_{i}" for i in range(5)]
    result = import_statement("from module", balanced_imports, config=balanced_config)
    lines = result.split("\n")
    if len(lines) > 1:
        line_lengths = [len(line) for line in lines[:-1]]
        assert max(line_lengths) - min(line_lengths) <= 1
    
    # Test single import that shouldn't wrap
    result = import_statement("from module", ["single_import"], config=custom_config)
    assert result.count("\n") == 0
    assert result == "from module import single_import"
    
    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
        config=custom_config,
    )
    assert result.count("\n") > 0
    
    # Test with empty imports list
    result = import_statement("from module", [])
    assert result == "from module"
    
    # Test with trailing comma config
    no_comma_config = Config(
        line_length=80,
        wrap_length=None,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=False,
        indent="    ",
        comment_prefix="  # ",
        ignore_comments=False,
        balanced_wrapping=False,
        use_parentheses=True,
    )
    
    result = import_statement("from module", ["import1", "import2", "import3"], config=no_comma_config)
    if result.count("\n") > 0:
        assert not result.strip().endswith(",")


# LLM-generated content at query #29
#--------------------------

```python
def test_import_statement():
    # Test basic single line import
    result = import_statement("from module", ["import1", "import2"])
    assert result == "from module import import1, import2"

    # Test with explode=True (vertical hanging indent)
    result = import_statement("from module", ["import1", "import2"], explode=True)
    assert result == "from module import (\n    import1,\n    import2,\n)"

    # Test with comments
    result = import_statement("from module", ["import1"], comments=["comment1"])
    assert result == "from module import import1  # comment1"

    # Test with multiple comments
    result = import_statement("from module", ["import1", "import2"], comments=["comment1", "comment2"])
    assert "import1  # comment1" in result
    assert "import2  # comment2" in result

    # Test with custom config
    custom_config = Config(
        line_length=50,
        wrap_length=40,
        indent="    ",
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        balanced_wrapping=False,
        comment_prefix="  # ",
        ignore_comments=False,
        use_parentheses=True,
    )
    result = import_statement(
        "from very_long_module_name",
        ["very_long_import_name1", "very_long_import_name2"],
        config=custom_config,
    )
    assert "from very_long_module_name import (" in result
    assert "very_long_import_name1," in result
    assert "very_long_import_name2," in result

    # Test with line_separator
    result = import_statement("from module", ["import1", "import2"], line_separator="\r\n", explode=True)
    assert "\r\n" in result
    assert result.endswith(",\r\n)")

    # Test with multi_line_output override
    result = import_statement(
        "from module",
        ["import1", "import2", "import3", "import4"],
        multi_line_output=Modes.VERTICAL_GRID_GROUPED,
    )
    assert "import1," in result
    assert "import2," in result

    # Test balanced_wrapping behavior
    balanced_config = Config(
        line_length=30,
        wrap_length=30,
        indent="    ",
        balanced_wrapping=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        include_trailing_comma=True,
    )
    result = import_statement(
        "from module",
        ["very_long_import_name1", "very_long_import_name2", "import3"],
        config=balanced_config,
    )
    lines = result.split("\n")
    assert len(lines) > 1

    # Test single import with explode
    result = import_statement("from module", ["single_import"], explode=True)
    assert result == "from module import (\n    single_import,\n)"

    # Test empty imports list
    result = import_statement("from module", [])
    assert result == "from module import "

    # Test with remove_comments=True in config
    no_comments_config = Config(ignore_comments=True)
    result = import_statement("from module", ["import1"], comments=["comment1"], config=no_comments_config)
    assert "# comment1" not in result
    assert "import1" in result

    # Test long import that should wrap
    long_config = Config(line_length=40, wrap_length=40, multi_line_output=Modes.VERTICAL_HANGING_INDENT)
    result = import_statement(
        "from very_long_module_name_here",
        ["import1", "import2", "import3", "import4", "import5"],
        config=long_config,
    )
    assert "(" in result
    assert "\n" in result
    assert result.endswith(")")

    # Test that trailing comma is included when configured
    trailing_comma_config = Config(
        include_trailing_comma=True,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=30,
    )
    result = import_statement("from module", ["import1", "import2"], config=trailing_comma_config)
    assert result.endswith(",\n)")

    # Test without trailing comma when configured
    no_trailing_comma_config = Config(
        include_trailing_comma=False,
        multi_line_output=Modes.VERTICAL_HANGING_INDENT,
        line_length=30,
    )
    result = import_statement("from module", ["import1", "import2"], config=no_trailing_comma_config)
    assert not result.endswith(",\n)")


