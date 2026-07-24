####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 2: No sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import print_function\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 3: Reverse sort
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "import sys\n"
        "import os\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 4: Star first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": ["*"],
                    "sys": ["path"],
                    "collections": ["defaultdict"]
                }
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from os import *\n"
        "from collections import defaultdict\n"
        "from sys import path\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 5: From first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 6: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():", "    pass"],
        imports={},
        import_index=-1,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "def foo():\n    pass\n"

    # Test case 7: Custom import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == (
        "# Third Party Imports\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 8: Lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import print_function\n"
        "\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 9: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )

    # Test case 10: Remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os", "collections"])
    result = sorted_imports(parsed, config)
    assert result == (
        "import sys\n"
        "\n"
        "def foo():\n"
        "    pass\n"
    )


# LLM-generated content at query #2
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == ""

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict", "defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict, defaultdict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 4: With sections and headings
    config = Config(
        import_headings={
            "thirdparty": "Third Party Imports",
            "future": "Future Imports",
        }
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            },
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Future Imports\n"
        "import __future__\n"
        "\n"
        "# Third Party Imports\n"
        "from collections import OrderedDict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = "import sys\n"
    assert result == expected

    # Test case 6: With lines_between_sections
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import __future__\n"
        "\n"
        "\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 7: With no_sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            },
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import __future__\n"
        "from collections import OrderedDict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 8: With force_sort_within_sections
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 9: With lines_after_imports
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 10: With place_imports
    config = DEFAULT_CONFIG
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"print('hello')": "THIRDPARTY"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "print('hello')\n"
        "import os\n"
        "\n"
    )
    assert result == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "os\nsys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 2: With sections and custom headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["models"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library"},
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config)
    assert "# Future imports" in result
    assert "# Standard library" in result
    assert "from __future__ import print_function" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from django import models" in result

    # Test case 3: Remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": [], "unused": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(remove_imports=["unused"])
    result = sorted_imports(parsed, config)
    assert "unused" not in result
    assert "os" in result
    assert "sys" in result

    # Test case 4: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"django": ["models"], "flask": ["Flask"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    os_index = lines.index("import os")
    zlib_index = lines.index("import zlib")
    assert os_index < zlib_index
    django_index = lines.index("from django import models")
    flask_index = lines.index("from flask import Flask")
    assert django_index < flask_index

    # Test case 5: Empty imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed)
    assert result == "x = 1\n"

    # Test case 6: Lines between types
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"django": ["models"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_between_types=2)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    import_index = lines.index("import os")
    from_index = lines.index("from django import models")
    assert from_index - import_index == 3  # 2 empty lines + 1 line

    # Test case 7: Star first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"django": ["*"], "flask": ["Flask"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    django_index = lines.index("from django import *")
    flask_index = lines.index("from flask import Flask")
    assert django_index < flask_index

    # Test case 8: From first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"django": ["models"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    from_index = lines.index("from django import models")
    import_index = lines.index("import os")
    assert from_index < import_index

    # Test case 9: Reverse sort
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"django": ["models"], "flask": ["Flask"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    sys_index = lines.index("import sys")
    os_index = lines.index("import os")
    assert sys_index < os_index
    flask_index = lines.index("from flask import Flask")
    django_index = lines.index("from django import models")
    assert flask_index < django_index

    # Test case 10: Custom formatting function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    assert "\r\n" in result


# LLM-generated content at query #4
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.path"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nfrom sys import sys.path\n\nprint('hello')\n"
    assert result == expected

    # Test case 3: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os.path"]}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["django.conf"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\n\nimport os\n\nfrom django import django.conf\n\nprint('hello')\n"
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.path"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        from_first=True,
        lines_between_sections=2,
        lines_between_types=1,
    )
    result = sorted_imports(parsed, config)
    expected = "\nfrom sys import sys.path\n\nimport os\n\nprint('hello')\n"
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.path"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = "\nfrom sys import sys.path\n\nprint('hello')\n"
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports in parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"collections": ["from collections import defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["from __future__ import annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"collections": ["from collections import defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With star_first config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["from module1 import *"],
                    "module2": ["from module2 import func"],
                },
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from module1 import *\n"
        "from module2 import func\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With import headings and footers
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "THIRD PARTY IMPORTS"},
        import_footers={"thirdparty": "END THIRD PARTY"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# THIRD PARTY IMPORTS\n"
        "import os\n"
        "\n"
        "# END THIRD PARTY\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 6: With remove_imports config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 7: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    expected = "import os\r\n\r\nprint('hello')\r\n"
    assert result == expected

    # Test case 8: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Place imports here", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"# Place imports here": "THIRDPARTY"},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "# Place imports here\n"
        "import os\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.path"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nfrom sys import sys.path\n\nprint('hello')"
    assert result == expected

    # Test case 3: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os.path"]}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["django.conf"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\n\nimport os\n\nfrom django import django.conf\n\nprint('hello')"
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nprint('hello')"
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os.path"]}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["django.conf"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\nimport os\nfrom django import django.conf\n\nprint('hello')"
    assert result == expected

    # Test case 6: With force_sort_within_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os.path"]},
                "from": {"sys": ["sys.path"], "django": ["django.conf"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport zlib\n\nfrom django import django.conf\nfrom sys import sys.path\n\nprint('hello')"
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n"

    # Test with no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom __future__ import absolute_import\n\nfrom collections import defaultdict\nimport os\nimport sys\n\n"

    # Test with force_sort_within_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n"

    # Test with from_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\n"

    # Test with star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"collections": ["*"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import *\nfrom itertools import chain\n\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Third Party Imports\nfrom collections import defaultdict\nimport os\nimport sys\n\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["os", "collections"])
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport sys\n\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom __future__ import absolute_import\n\n\n\nfrom collections import defaultdict\nimport os\nimport sys\n\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nimport os\nimport sys\n\n\ndef foo():\n    pass\n"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\ndef foo():\n    pass\n\nfrom collections import defaultdict\nimport os\nimport sys\n\n"


# LLM-generated content at query #8
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"collections": ["from collections import defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["from __future__ import annotations"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["import os"]},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"future": "Future", "stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    expected = (
        "# Future\n"
        "from __future__ import annotations\n"
        "\n"
        "# Standard Library\n"
        "import os\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 4: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"django": ["import django"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"myapp": ["import myapp"]},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed.imports["LOCALFOLDER"] = {"straight": {"local": ["import local"]}, "from": {}}
    result = sorted_imports(parsed, config)
    expected = (
        "import django\n"
        "\n"
        "import local\n"
        "\n"
        "import myapp\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["from __future__ import annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"collections": ["from collections import defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')"
    )
    assert result == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == ""

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 4: From imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": ["path"],
                    "sys": ["argv"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nfrom os import path\nfrom sys import argv\n\nprint('hello')\n"
    assert result == expected

    # Test case 5: Mixed imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nfrom sys import argv\n\nprint('hello')\n"
    assert result == expected

    # Test case 6: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"sys": ["argv"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import __future__\n\nimport os\nfrom sys import argv\n\nprint('hello')\n"
    assert result == expected

    # Test case 7: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"sys": ["argv"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import __future__\n\nimport os\nfrom sys import argv\n\nprint('hello')\n"
    assert result == expected

    # Test case 8: With remove_imports config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nprint('hello')\n"
    assert result == expected

    # Test case 9: With lines_between_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import __future__\n\n\n\nimport os\n\nprint('hello')\n"
    assert result == expected

    # Test case 10: With import_headings config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = "\n# Third Party Imports\nimport os\n\nprint('hello')\n"
    assert result == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\n\nfrom collections import defaultdict\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import defaultdict\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\nimport os\n\nfrom collections import defaultdict\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n"


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n"

    # Test case 2: No sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\nimport sys\n\nx = 1\n"

    # Test case 3: Empty imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 4: Star imports first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func1"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import func1\n\nx = 1\n"

    # Test case 5: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n"


# LLM-generated content at query #12
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 2: Basic imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict", "OrderedDict"}},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    expected = "\n".join([
        "import os",
        "import sys",
        "",
        "from collections import OrderedDict, defaultdict",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 3: With sections and headings
    config = Config(
        import_headings={
            "thirdparty": "Third Party Imports",
            "firstparty": "First Party Imports",
        },
        lines_between_sections=1,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"requests": set()},
                "from": {"flask": {"Flask"}},
            },
            "FIRSTPARTY": {
                "straight": {"myapp": set()},
                "from": {"myapp.utils": {"helper"}},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "# Third Party Imports",
        "import requests",
        "",
        "from flask import Flask",
        "",
        "",
        "# First Party Imports",
        "import myapp",
        "",
        "from myapp.utils import helper",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With remove_imports
    config = Config(remove_imports=["import os", "from sys import *"])
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": set(), "sys": set()},
                "from": {"sys": {"*"}},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "import sys",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With no_sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "from __future__ import print_function",
        "",
        "import os",
        "",
        "from sys import exit",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 6: With star_first
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "numpy": {"*"},
                    "pandas": {"DataFrame"},
                },
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "from numpy import *",
        "from pandas import DataFrame",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 7: With force_sort_within_sections
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"django": set(), "flask": set()},
                "from": {"requests": {"get"}, "urllib": {"request"}},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "from requests import get",
        "from urllib import request",
        "",
        "import django",
        "import flask",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 8: With formatting_function
    def custom_formatter(code, extension, config):
        return code.replace("import", "from").replace("from", "import")

    config = Config(formatting_function=custom_formatter)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "from os",
        "",
        "import sys import exit",
        "",
        "print('hello')",
        "",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 9: With lines_before_imports and lines_after_imports
    config = Config(lines_before_imports=2, lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n".join([
        "",
        "",
        "import os",
        "",
        "",
        "print('hello')",
    ])
    assert sorted_imports(parsed, config) == expected

    # Test case 10: With place_imports
    config = DEFAULT_CONFIG
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "# IMPORTS HERE"],
        imports={
            "STDLIB": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        place_imports={"STDLIB": ["import os"]},
        import_placements={"# IMPORTS HERE": "STDLIB"},
    )
    expected = "\n".join([
        "print('hello')",
        "# IMPORTS HERE",
        "import os",
        "",
    ])
    assert sorted_imports(parsed, config) == expected


# LLM-generated content at query #13
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        sections=[],
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')"
    )

    # Test case 3: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(force_sort_within_sections=True, lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "\n"
        "print('hello')"
    )

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        sections=["THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == (
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')"
    )

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        sections=["FUTURE", "THIRDPARTY"],
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import annotations\n"
        "import os\n"
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')"
    )


# LLM-generated content at query #14
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with force_sort_within_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"collections": ["*"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import *\nfrom itertools import chain\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n\nx = 1\n"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\nx = 1\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport sys\n"

    # Test with reverse_sort
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    assert result == "from itertools import chain\nfrom collections import defaultdict\nimport sys\nimport os\n"


# LLM-generated content at query #15
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"future": "Future imports", "stdlib": "Standard library imports"})
    result = sorted_imports(parsed, config)
    assert "# Future imports" in result
    assert "# Standard library imports" in result
    assert "from __future__ import annotations" in result
    assert "import os\nimport sys" in result

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"django": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"myapp": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert "import django" in result
    assert "import myapp" in result

    # Test case 5: With no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"myapp": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "import myapp" in result

    # Test case 6: With custom formatting function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    assert "IMPORT OS" in result
    assert "PRINT('HELLO')" in result

    # Test case 7: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "# Placeholder" in result
    assert "import os" in result
    assert "print('hello')" in result

    # Test case 8: With lines_before_imports and lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(lines_before_imports=2, lines_after_imports=2)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    assert lines[0] == ""
    assert lines[1] == ""
    assert lines[2] == "import os"
    assert lines[3] == ""
    assert lines[4] == ""
    assert lines[5] == "print('hello')"

    # Test case 9: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func1"], "module3": ["*"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert "from module1 import *" in result
    assert "from module3 import *" in result
    assert "from module2 import func1" in result

    # Test case 10: With ensure_newline_before_comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Comment", "print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "\n\n# Comment" in result
    assert "print('hello')" in result


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom __future__ import absolute_import\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["function"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom module1 import *\nfrom module2 import function\n\nx = 1\n"

    # Test with force_sort_within_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Third Party Imports\nimport os\n\nx = 1\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import absolute_import\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1"

    # Test with from_first=True
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first == "from collections import defaultdict\n\nimport os\n\nx = 1"

    # Test with star_first=True
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": {"*"},
                    "collections": {"defaultdict"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first == "from os import *\nfrom collections import defaultdict\n\nx = 1"

    # Test with import headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings == "# Third Party Imports\nimport os\n\nx = 1"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result_no_imports = sorted_imports(parsed_no_imports)
    assert result_no_imports == "x = 1"


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 3: With sections and comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Main code", "print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os.path"]}, "from": {"sys": ["sys.argv"]}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["models"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library"},
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Future imports\n"
        "from __future__ import annotations\n"
        "\n"
        "# Standard library\n"
        "import os\n"
        "\n"
        "from sys import sys.argv\n"
        "\n"
        "from django import models\n"
        "\n"
        "# Main code\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": ["array"]}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": ["func"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"], lines_between_sections=1)
    result = sorted_imports(parsed, config)
    expected = (
        "import numpy\n"
        "\n"
        "import my_module\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 5: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": ["array"]}, "from": {"django": ["models"]}},
            "FIRSTPARTY": {"straight": {"my_module": ["func"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import my_module\n"
        "import numpy\n"
        "\n"
        "from django import models\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 6: With star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "numpy": ["array", "*"],
                    "django": ["models"],
                },
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from numpy import *\n"
        "from django import models\n"
        "from numpy import array\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 7: With from_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["array"]},
                "from": {"django": ["models"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from django import models\n"
        "\n"
        "import numpy\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 8: With force_sort_within_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["array"], "pandas": ["DataFrame"]},
                "from": {"django": ["models"], "flask": ["Flask"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import numpy\n"
        "import pandas\n"
        "\n"
        "from django import models\n"
        "from flask import Flask\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 9: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": ["array"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import numpy\n"
        "\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Header", "print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": ["array"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        place_imports={"THIRDPARTY": ["import numpy"]},
        import_placements={"# Header": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "# Header\n"
        "import numpy\n"
        "\n"
        "print('hello')"
    )
    assert result == expected


# LLM-generated content at query #4
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"numpy": ["numpy as np"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys\n\nfrom numpy import numpy as np\n\nx = 1" == result

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert "x = 1" == result_no_imports

    # Test with config options
    config_custom = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
    result_custom = sorted_imports(parsed, config_custom)
    assert "from numpy import numpy as np\n\nimport sys\nimport os\n\n\nx = 1" == result_custom

    # Test with forced_separate
    config_forced = Config(forced_separate=["FORCED"])
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {}
            },
            "FORCED": {
                "straight": {"sys": ["sys"]},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "import os\n\nimport sys\n\nx = 1" == result_forced

    # Test with no_sections
    config_no_sections = Config(no_sections=True)
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"numpy": ["numpy as np"]}
            },
            "FIRSTPARTY": {
                "straight": {"my_module": ["my_module"]},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "import os\nimport sys\nimport my_module\n\nfrom numpy import numpy as np\n\nx = 1" == result_no_sections


# LLM-generated content at query #5
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        import_headings={"thirdparty": "Third Party"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party\n"
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["FUTURE"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "\nos\nsys\n\nfrom collections import defaultdict, OrderedDict\n\nx = 1"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "x = 1"

    # Test with forced sections
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}},
            "THIRDPARTY": {"straight": {"os": ["os"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_forced = Config(forced_separate=["FUTURE"], no_sections=False)
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert result_forced == "\nfrom __future__ import print_function\n\nos\n\nx = 1"

    # Test with custom section headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_headings = Config(
        import_headings={"thirdparty": "Third-party imports"},
        dedup_headings=True,
    )
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings == "\n# Third-party imports\nos\n\nx = 1"

    # Test with lines between sections
    parsed_lines = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "STDLIB": {"straight": {"os": ["os"]}},
            "THIRDPARTY": {"straight": {"django": ["django"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines == "\nos\n\n\ndjango\n\nx = 1"

    # Test with reverse sort
    parsed_reverse = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_reverse = Config(reverse_sort=True)
    result_reverse = sorted_imports(parsed_reverse, config_reverse)
    assert result_reverse == "\nsys\nos\n\nx = 1"

    # Test with star_first
    parsed_star = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "from": {
                    "module1": ["*"],
                    "module2": ["func1"],
                }
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star == "\nfrom module1 import *\nfrom module2 import func1\n\nx = 1"

    # Test with formatting function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed_format = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_format = Config(formatting_function=custom_format)
    result_format = sorted_imports(parsed_format, config_format)
    assert result_format == "\r\nos\r\n\r\nx = 1"

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["", "x = 1", "# IMPORT HERE"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}},
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["os"]},
        import_placements={"# IMPORT HERE": "THIRDPARTY"},
    )
    config_place = Config()
    result_place = sorted_imports(parsed_place, config_place)
    assert result_place == "\nx = 1\n# IMPORT HERE\nos\n"


# LLM-generated content at query #2
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        imports={},
        place_imports={},
        import_placements={},
    )
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys\nfrom collections import defaultdict\n\nprint('hello')\n" == result

    # Test case 3: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"sys": []}, "from": {"collections": ["defaultdict"]}},
        },
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert (
        "from __future__ import annotations\n\nimport os\n\nimport sys\nfrom collections import defaultdict\n\nprint('hello')\n"
        == result
    )

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"sys": [], "os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(
        reverse_sort=True,
        from_first=True,
        lines_between_sections=2,
        lines_between_types=1,
    )
    result = sorted_imports(parsed, config)
    assert (
        "from collections import defaultdict\n\nimport sys\nimport os\n\nprint('hello')\n"
        == result
    )

    # Test case 5: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        imports={
            "THIRDPARTY": {
                "straight": {"sys": [], "os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["SEPARATE"])
    parsed.imports["SEPARATE"] = {"straight": {"numpy": []}, "from": {}}
    result = sorted_imports(parsed, config)
    assert (
        "import numpy\n\nimport os\nimport sys\nfrom collections import defaultdict\n\nprint('hello')\n"
        == result
    )


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\n\nfrom collections import defaultdict\n"

    # Test with forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["THIRDPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import defaultdict\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"collections": ["*"], "os": ["path"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import *\nfrom os import path\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n"

    # Test with import_footers
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import defaultdict\n\n# End of Third Party Imports\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["defaultdict"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\n\nimport os\n\nfrom collections import defaultdict\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():\n    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\ndef foo():\n    pass\n"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():\n    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\ndef foo():\n    pass\n"

    # Test with formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=lambda x, y, z: x.replace("\n", "\r\n"))
    result = sorted_imports(parsed, config)
    assert result == "import os\r\n"

    # Test with place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "def foo():\n    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import sys"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n# Placeholder\nimport sys\n\ndef foo():\n    pass\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():\n    pass"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "def foo():\n    pass\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "\n"

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"], "typing": ["List"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict\n"
        "from typing import List\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With config options
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.from_first = True
    config.lines_between_sections = 2
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {"sys": ["exit"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "STDLIB"],
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import print_function\n"
        "\n"
        "\n"
        "from sys import exit\n"
        "\n"
        "import os\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.remove_imports = ["import os"]
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["STDLIB"],
    )
    result = sorted_imports(parsed, config)
    expected = "import sys\n\n"
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test with no_sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "import __future__" in result_no_sections
    assert "import os" in result_no_sections
    assert "import sys" in result_no_sections

    # Test with only_sections
    parsed_only_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_only_sections = Config(only_sections=True)
    result_only_sections = sorted_imports(parsed_only_sections, config_only_sections)
    assert "import os" in result_only_sections
    assert "import sys" in result_only_sections
    assert "from collections import defaultdict" in result_only_sections
    assert "from itertools import chain" in result_only_sections

    # Test with star_first
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["function"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first.index("from module1 import *") < result_star_first.index(
        "from module2 import function"
    )

    # Test with from_first
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from collections import defaultdict") < result_from_first.index(
        "import os"
    )

    # Test with import_headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings
    assert result_headings.index("# Third Party Imports") < result_headings.index("import os")

    # Test with import_footers
    parsed_footers = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_footers = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result_footers = sorted_imports(parsed_footers, config_footers)
    assert "# End of Third Party Imports" in result_footers
    assert result_footers.index("# End of Third Party Imports") > result_footers.index("import os")

    # Test with lines_between_sections
    parsed_lines_between = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_lines_between = Config(lines_between_sections=2)
    result_lines_between = sorted_imports(parsed_lines_between, config_lines_between)
    assert result_lines_between.count("") >= 2

    # Test with lines_between_types
    parsed_lines_between_types = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_lines_between_types = Config(lines_between_types=1)
    result_lines_between_types = sorted_imports(parsed_lines_between_types, config_lines_between_types)
    assert result_lines_between_types.count("") >= 1

    # Test with force_sort_within_sections
    parsed_force_sort = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_force_sort, config_force_sort)
    assert "import os" in result_force_sort
    assert "import sys" in result_force_sort
    assert "from collections import defaultdict" in result_force_sort
    assert "from itertools import chain" in result_force_sort

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_remove = Config(remove_imports=["os", "collections"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" not in result_remove
    assert "from collections import defaultdict" not in result_remove
    assert "import sys" in result_remove
    assert "from itertools import chain" in result_remove


# LLM-generated content at query #6
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        imports={},
        place_imports={},
        import_placements={},
        original_line_count=1,
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = """import os
import sys

from collections import defaultdict

print('hello')"""
    assert result == expected

    # Test case 3: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
        sections=["THIRDPARTY"],
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = """from collections import defaultdict

import os
import sys

print('hello')"""
    assert result == expected

    # Test case 4: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
        sections=["THIRDPARTY"],
    )
    config = Config(forced_separate=["numpy"])
    result = sorted_imports(parsed, config)
    expected = """import os
import sys

from collections import defaultdict

print('hello')"""
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            },
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = """from __future__ import annotations
import os
import sys

from collections import defaultdict

print('hello')"""
    assert result == expected


# LLM-generated content at query #7
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "\n"

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "print('hello')\n"

    # Test case 3: Simple imports sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert "import os\n" in result
    assert "import sys\n" in result
    assert "from collections import defaultdict\n" in result

    # Test case 4: With sections and comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Main code"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}},
            "STDLIB": {"straight": {"os": ["path"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library"},
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config)
    assert "# Future imports" in result
    assert "# Standard library" in result
    assert "import __future__\n" in result
    assert "import os\n" in result

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"django": ["models"]}},
            "FIRSTPARTY": {"straight": {"myapp": ["utils"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert "import django\n" in result
    assert "import myapp\n" in result

    # Test case 6: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}},
            "STDLIB": {"straight": {"os": ["path"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "# Future imports" in result
    assert "import __future__\n" in result
    assert "import os\n" in result

    # Test case 7: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["array"], "pandas": ["DataFrame"]},
                "from": {"typing": ["List"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["from typing import List"])
    result = sorted_imports(parsed, config)
    assert "import numpy\n" in result
    assert "import pandas\n" in result
    assert "from typing import List" not in result

    # Test case 8: With formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"THIRDPARTY": {"straight": {"black": ["format_file"]}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = sorted_imports(parsed, config)
    assert "IMPORT BLACK" in result

    # Test case 9: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "def main():", "    pass"],
        imports={"THIRDPARTY": {"straight": {"requests": ["get"]}}},
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import requests"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    result = sorted_imports(parsed)
    assert "import requests\n" in result
    assert "# Placeholder" in result
    assert "def main():\n" in result

    # Test case 10: With lines_after_imports and lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def foo():", "    pass"],
        imports={"STDLIB": {"straight": {"sys": ["exit"]}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2, lines_after_imports=2)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    assert lines[0] == ""
    assert lines[1] == ""
    assert lines[2] == "import sys"
    assert lines[3] == ""
    assert lines[4] == ""
    assert lines[5] == "def foo():"


# LLM-generated content at query #8
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"future": "Future imports", "thirdparty": "Third party imports"})
    result = sorted_imports(parsed, config)
    expected = (
        "# Future imports\n"
        "from __future__ import print_function\n"
        "\n"
        "# Third party imports\n"
        "import os\n"
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import print_function\n"
        "import collections\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result.startswith("import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef main():")

    # Test with no sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections.startswith("from __future__ import print_function\n\nimport os\nimport sys\n\n\ndef main():")

    # Test with forced separate
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_forced = Config(forced_separate=["FIRSTPARTY"])
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert result_forced.startswith("import os\nimport sys\n\nimport my_module\n\n\ndef main():")

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_remove = Config(remove_imports=["os"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert result_remove.startswith("import sys\n\n\ndef main():")

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=2,
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "def main():\n    pass"

    # Test with star_first
    parsed_star = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func1", "func2"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star.startswith("from module1 import *\nfrom module2 import func1, func2\n\n\ndef main():")

    # Test with from_first
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.startswith("from collections import defaultdict\n\nimport os\nimport sys\n\n\ndef main():")

    # Test with lines_between_sections
    parsed_lines = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines.startswith("from __future__ import print_function\n\n\n\nimport os\n\n\ndef main():")

    # Test with import_headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings.startswith("# Third Party Imports\nimport os\n\n\ndef main():")

    # Test with ensure_newline_before_comments
    parsed_comments = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_comments = Config(ensure_newline_before_comments=True)
    result_comments = sorted_imports(parsed_comments, config_comments)
    assert result_comments.startswith("import os\n\n\ndef main():")


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == ""

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=[],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["path"], "collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from collections import defaultdict\n"
        "from sys import path"
    )
    assert result == expected

    # Test case 4: With config options
    config = Config(
        force_sort_within_sections=True,
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["path"], "collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "from sys import path\n"
        "\n"
        "import os\n"
        "import zlib"
    )
    assert result == expected

    # Test case 5: With import headings
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["path"], "collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import os\n"
        "import zlib\n"
        "\n"
        "from collections import defaultdict\n"
        "from sys import path"
    )
    assert result == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import my_module\nimport os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with force_sort_within_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\nimport sys\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"collections": ["*"], "os": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import *\nfrom os import path\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\n\nimport my_module\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\ndef main():    pass\n"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\ndef main():    pass\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nfrom collections import defaultdict\n"

    # Test with formatting_function
    def custom_formatter(code, extension, config):
        return code.replace("import", "from").replace("from", "import")

    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=custom_formatter)
    result = sorted_imports(parsed, config)
    assert result == "from os\n\nimport collections import defaultdict\n"


# LLM-generated content at query #12
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport os\nimport sys\n"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\nimport os\n"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": None}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nimport my_module\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport sys\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["*"],
                    "module2": ["function1"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import function1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": None}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nimport my_module\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\ndef main():\n    pass\n"

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": None}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    assert result == "import os\r\n"


# LLM-generated content at query #13
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports in parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import Counter, defaultdict\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 3: With sections and custom config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["models"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library"},
        lines_between_sections=2,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Future imports\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "# Standard library\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from django import models\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os", "collections.defaultdict"])
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 5: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
            "THIRDPARTY": {"straight": {"django": ["django"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "import django\n"
        "import os\n"
        "\n"
        "print('hello')"
    )
    assert result == expected


# LLM-generated content at query #14
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import absolute_import\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with from_first=True
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed, config_from_first)
    assert result_from_first == "from json import load\nfrom sys import path\n\nimport os\nimport zlib\n"

    # Test with star_first=True
    parsed_star = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"sys": ["*"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star, config_star_first)
    assert result_star_first == "from sys import *\nfrom json import load\n"

    # Test with import_headings
    config_headings = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    result_headings = sorted_imports(parsed, config_headings)
    assert result_headings == "# Third Party Imports\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with lines_between_sections
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed, config_lines)
    assert result_lines == "import os\nimport zlib\n\n\nfrom json import load\nfrom sys import path\n"

    # Test with force_sort_within_sections
    parsed_comments = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_comments, config_force_sort)
    assert result_force_sort == "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with ensure_newline_before_comments
    parsed_comments = parse.ParsedContent(
        lines_without_imports=["# Comment"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_newline = Config(ensure_newline_before_comments=True)
    result_newline = sorted_imports(parsed_comments, config_newline)
    assert result_newline == "import os\n\n# Comment\n"

    # Test with remove_imports
    config_remove = Config(remove_imports=["os"])
    result_remove = sorted_imports(parsed, config_remove)
    assert result_remove == "import zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with lines_after_imports
    parsed_with_code = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_lines_after = Config(lines_after_imports=2)
    result_lines_after = sorted_imports(parsed_with_code, config_lines_after)
    assert result_lines_after == "import os\n\n\nx = 1\n"

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["# Placeholder"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import sys"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config_place = Config()
    result_place = sorted_imports(parsed_place, config_place)
    assert result_place == "import os\n# Placeholder\nimport sys\n"


# LLM-generated content at query #15
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import annotations\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with from_first=True
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["argv"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first == "from sys import argv\nimport os\n\nx = 1\n"

    # Test with star_first=True
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["*"],  # star import
                    "sys": ["argv"],  # regular import
                }
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first == "from os import *\nfrom sys import argv\n\nx = 1\n"

    # Test with force_sort_within_sections=True
    parsed_force_sort = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_force_sort, config_force_sort)
    assert result_force_sort == "import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n"

    # Test with custom import_headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings == "# Third Party Imports\nimport os\n\nx = 1\n"

    # Test with lines_between_sections
    parsed_lines_between = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_lines_between = Config(lines_between_sections=2)
    result_lines_between = sorted_imports(parsed_lines_between, config_lines_between)
    assert result_lines_between == "from __future__ import annotations\n\n\nimport os\n\nx = 1\n"

    # Test with lines_after_imports
    parsed_lines_after = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config_lines_after = Config(lines_after_imports=2)
    result_lines_after = sorted_imports(parsed_lines_after, config_lines_after)
    assert result_lines_after == "import os\n\n\nx = 1\n"

    # Test with ensure_newline_before_comments=True
    parsed_newline_before_comments = parse.ParsedContent(
        lines_without_imports=["", "x = 1", "# comment"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config_newline_before_comments = Config(ensure_newline_before_comments=True)
    result_newline_before_comments = sorted_imports(
        parsed_newline_before_comments, config_newline_before_comments
    )
    assert result_newline_before_comments == "import os\n\nx = 1\n\n# comment\n"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
    )
    result_no_imports = sorted_imports(parsed_no_imports)
    assert result_no_imports == "x = 1\n"


