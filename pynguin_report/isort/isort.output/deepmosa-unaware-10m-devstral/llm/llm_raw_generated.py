####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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

    # Test case 2: No imports
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

    # Test case 3: With sections and comments
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {"sys": ["argv"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(import_headings={"future": "Future", "stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Future\nfrom __future__ import annotations\n\n# Standard Library\nimport os\n\nfrom sys import argv\n\nx = 1\n"

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"django": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"myapp": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed.imports["LOCALFOLDER"] = {"straight": {".utils": []}, "from": {}}
    result = sorted_imports(parsed, config)
    assert ".utils" in result
    assert "django" in result
    assert "myapp" in result

    # Test case 5: With remove_imports
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


# LLM-generated content at query #2
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["import zlib"], "os": ["import os"]},
                "from": {"sys": ["from sys import path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom sys import path\n\nx = 1\n"

    # Test with no_sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["from __future__ import annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"zlib": ["import zlib"], "os": ["import os"]},
                "from": {"sys": ["from sys import path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import annotations\nimport os\nimport zlib\n\nfrom sys import path\n\nx = 1\n"

    # Test with force_sort_within_sections
    parsed_force_sort = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["import zlib"], "os": ["import os"]},
                "from": {"sys": ["from sys import path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_force_sort, config_force_sort)
    assert result_force_sort == "import os\nimport zlib\n\nfrom sys import path\n\nx = 1\n"

    # Test with from_first
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["import zlib"], "os": ["import os"]},
                "from": {"sys": ["from sys import path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first == "from sys import path\n\nimport os\nimport zlib\n\nx = 1\n"

    # Test with star_first
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["import zlib"], "os": ["import os"]},
                "from": {"sys": ["from sys import *"], "os": ["from os import path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first == "import os\nimport zlib\n\nfrom sys import *\nfrom os import path\n\nx = 1\n"


# LLM-generated content at query #3
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
    assert sorted_imports(parsed, config) == "\n"

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
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 3: Basic sorting with default config
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
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    expected = "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With custom config (reverse_sort=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "zlib": []},
                "from": {"json": ["load"], "sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(reverse_sort=True)
    expected = "import zlib\nimport os\n\nfrom sys import path\nfrom json import load\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With forced_separate config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {"my_package": ["utils"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    expected = "import os\n\nfrom sys import path\n\nimport my_module\n\nfrom my_package import utils\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 6: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    expected = "from __future__ import annotations\nimport os\nimport zlib\n\nfrom sys import path\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 7: With star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": ["*"],  # star import
                    "sys": ["path"],
                    "json": ["load"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    expected = "from os import *\nfrom json import load\nfrom sys import path\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 8: With from_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "zlib": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    expected = "from json import load\nfrom sys import path\n\nimport os\nimport zlib\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 9: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    expected = "# Third Party Imports\nimport os\n\nfrom sys import path\n\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 10: With lines_between_sections=2
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
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
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    expected = "import os\n\n\nimport my_module\n\n"
    assert sorted_imports(parsed, config) == expected


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
    config = Config()
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["sys"], "json": ["json"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport zlib\n\nfrom json import json\nfrom sys import sys\n"
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["sys"], "json": ["json"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(lines_between_types=1, from_first=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom json import json\nfrom sys import sys\n\nimport os\nimport zlib\n"
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["sys"], "json": ["json"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(remove_imports=["zlib", "json"])
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nfrom sys import sys\n"
    assert result == expected

    # Test case 6: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["sys"], "json": ["json"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = "\n# Third Party Imports\nimport os\nimport zlib\n\nfrom json import json\nfrom sys import sys\n"
    assert result == expected

    # Test case 7: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def func():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["sys"], "json": ["json"]},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os", "import zlib"]},
        import_placements={"def func():"},
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "\n\ndef func():\nimport os\nimport zlib\n    pass\n"
    assert result == expected


# LLM-generated content at query #5
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"STDLIB": {"straight": {}, "from": {}}},
        import_index=0,
        line_separator="\n",
    )
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"os.path": ["from os.path import join"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport os\nimport sys\n\nfrom os.path import join\n\nprint('hello')"
    assert sorted_imports(parsed) == expected

    # Test case 4: With config modifications
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"sys": ["import sys"], "os": ["import os"]},
                "from": {"os.path": ["from os.path import join"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport os\nimport sys\n\nfrom os.path import join\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With sections and headings
    config = Config(import_headings={"stdlib": "Standard Library"})
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\n# Standard Library\nimport os\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With forced separate sections
    config = Config(forced_separate=["THIRDPARTY"])
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": ["import numpy"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport os\n\nimport numpy\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With remove_imports
    config = Config(remove_imports=["import os"])
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport sys\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 8: With lines_between_sections
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": ["import numpy"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport os\n\n\n\nimport numpy\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 9: With lines_after_imports
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\nimport os\n\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 10: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    config = Config(formatting_function=custom_format)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
    )
    expected = "\r\nimport os\r\n\r\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected


# LLM-generated content at query #6
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        imports={},
        place_imports={},
        import_placements={},
        original_line_count=1,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"datetime": ["from datetime import datetime"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nfrom datetime import datetime\n\nprint('hello')"
    assert result == expected

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "FUTURE": {
                "straight": {"__future__": ["from __future__ import annotations"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"datetime": ["from datetime import datetime"]},
            },
            "THIRDPARTY": {
                "straight": {"numpy": ["import numpy"], "pandas": ["import pandas"]},
                "from": {"django": ["from django.conf import settings"]},
            },
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(
        import_headings={"future": "Future", "stdlib": "Standard Library", "thirdparty": "Third Party"},
        dedup_headings=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "\n# Future\nfrom __future__ import annotations\n\n# Standard Library\nimport os\nimport sys\n\nfrom datetime import datetime\n\n# Third Party\nimport numpy\nimport pandas\n\nfrom django.conf import settings\n\nprint('hello')"
    )
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"datetime": ["from datetime import datetime"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["import os"])
    result = sorted_imports(parsed, config)
    expected = "\nimport sys\n\nfrom datetime import datetime\n\nprint('hello')"
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "FUTURE": {
                "straight": {"__future__": ["from __future__ import annotations"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"datetime": ["from datetime import datetime"]},
            },
            "THIRDPARTY": {
                "straight": {"numpy": ["import numpy"], "pandas": ["import pandas"]},
                "from": {"django": ["from django.conf import settings"]},
            },
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "\nfrom __future__ import annotations\nimport numpy\nimport os\nimport pandas\nimport sys\n\nfrom datetime import datetime\nfrom django.conf import settings\n\nprint('hello')"
    )
    assert result == expected


# LLM-generated content at query #7
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
                "from": {"collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"

    # Test case 3: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "STDLIB": {"straight": {"os": set()}, "from": {}},
            "THIRDPARTY": {"straight": {"django": set()}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\nimport os\n\nimport django\n\nprint('hello')\n"

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(from_first=True, lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nfrom collections import defaultdict\n\nprint('hello')\n"


# LLM-generated content at query #8
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\nimport sys\n\nx = 1\n"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nimport sys\n\nx = 1\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    assert result == "import sys\n\nfrom collections import defaultdict\n\nx = 1\n"

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

    # Test with import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nx = 1\n"

    # Test with lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nimport sys\n\nx = 1\n"

    # Test with lines after imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nx = 1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["defaultdict"],
                    "os": ["*"],
                },
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from os import *\nfrom collections import defaultdict\n\nx = 1\n"


# LLM-generated content at query #9
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
    config = Config()
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Basic sorting with default config
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
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict, defaultdict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 4: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import __future__\n\n"
        "\n"
        "from collections import defaultdict\n"
        "import os\n"
    )
    assert result == expected

    # Test case 5: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "from collections import defaultdict\n"
        "import os\n"
    )
    assert result == expected

    # Test case 6: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            },
            "FIRSTPARTY": {"straight": {"my_module": ["my_module"]}, "from": {}},
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "import os\n"
        "\n"
        "import my_module\n"
    )
    assert result == expected

    # Test case 7: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import __future__\n"
        "from collections import defaultdict\n"
        "import os\n"
    )
    assert result == expected

    # Test case 8: With star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["*"],
                    "module2": ["func1", "func2"],
                    "module3": ["*"],
                },
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from module1 import *\n"
        "from module3 import *\n"
        "from module2 import func1, func2\n"
    )
    assert result == expected

    # Test case 9: With from_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
    )
    assert result == expected

    # Test case 10: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "\n"
        "def main():\n"
        "    pass\n"
    )
    assert result == expected


# LLM-generated content at query #10
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result.startswith("import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1")

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "x = 1"

    # Test with forced_separate
    config_forced = Config(forced_separate=["numpy"])
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            },
            "numpy": {
                "straight": {"numpy": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "numpy" in result_forced

    # Test with no_sections
    config_no_sections = Config(no_sections=True)
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": []},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "from __future__" in result_no_sections
    assert "import os" in result_no_sections

    # Test with remove_imports
    config_remove = Config(remove_imports=["zlib"])
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "zlib" not in result_remove
    assert "os" in result_remove

    # Test with star_first
    config_star = Config(star_first=True)
    parsed_star = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["path"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star.index("from os import *") < result_star.index("from sys import path")

    # Test with from_first
    config_from_first = Config(from_first=True)
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from sys import path") < result_from_first.index("import os")

    # Test with import_headings
    config_headings = Config(import_headings={"thirdparty": "Third Party"})
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party" in result_headings

    # Test with lines_between_sections
    config_lines = Config(lines_between_sections=2)
    parsed_lines = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": []},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines.count("\n\n") >= 2

    # Test with lines_after_imports
    config_after = Config(lines_after_imports=2)
    parsed_after = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_after = sorted_imports(parsed_after, config_after)
    assert result_after.count("\nimport os\n\n\n\nx = 1") == 1

    # Test with ensure_newline_before_comments
    config_newline = Config(ensure_newline_before_comments=True)
    parsed_newline = parse.ParsedContent(
        lines_without_imports=["# comment", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    result_newline = sorted_imports(parsed_newline, config_newline)
    assert "\n\n# comment" in result_newline


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed_content = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed_content, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n"

    # Test with no_sections=True
    parsed_content_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_content_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import absolute_import\nfrom collections import defaultdict\nimport os\nimport sys\n\nx = 1\n"

    # Test with from_first=True
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_content, config_from_first)
    assert result_from_first == "from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with star_first=True
    parsed_content_star = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_content_star, config_star_first)
    assert result_star_first == "from module1 import *\nfrom module2 import func\n\nx = 1\n"

    # Test with lines_between_types
    config_lines_between = Config(lines_between_types=2)
    result_lines_between = sorted_imports(parsed_content, config_lines_between)
    assert result_lines_between == "from collections import defaultdict\nfrom itertools import chain\n\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with lines_between_sections
    parsed_content_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_lines_between_sections = Config(lines_between_sections=2)
    result_lines_between_sections = sorted_imports(parsed_content_sections, config_lines_between_sections)
    assert result_lines_between_sections == "from __future__ import absolute_import\n\n\nimport os\n\nx = 1\n"

    # Test with import_headings
    config_headings = Config(import_headings={"thirdparty": "Third Party"})
    result_headings = sorted_imports(parsed_content, config_headings)
    assert result_headings == "# Third Party\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n"

    # Test with dedup_headings
    config_dedup = Config(import_headings={"thirdparty": "Third Party"}, dedup_headings=True)
    parsed_content_dedup = parse.ParsedContent(
        lines_without_imports=["# Third Party", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    result_dedup = sorted_imports(parsed_content_dedup, config_dedup)
    assert result_dedup == "# Third Party\nfrom collections import defaultdict\nimport os\n\nx = 1\n"

    # Test with lines_after_imports
    config_lines_after = Config(lines_after_imports=2)
    result_lines_after = sorted_imports(parsed_content, config_lines_after)
    assert result_lines_after == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n\nx = 1\n"

    # Test with lines_before_imports
    config_lines_before = Config(lines_before_imports=2)
    result_lines_before = sorted_imports(parsed_content, config_lines_before)
    assert result_lines_before == "\n\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n"

    # Test with force_sort_within_sections
    parsed_content_force_sort = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_content_force_sort, config_force_sort)
    assert result_force_sort == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n"

    # Test with remove_imports
    config_remove = Config(remove_imports=["os"])
    result_remove = sorted_imports(parsed_content, config_remove)
    assert result_remove == "from collections import defaultdict\nfrom itertools import chain\nimport sys\n\nx = 1\n"

    # Test with only_sections
    config_only_sections = Config(only_sections=True)
    result_only_sections = sorted_imports(parsed_content, config_only_sections)
    assert result_only_sections == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\nx = 1\n"

    # Test with reverse_sort
    config_reverse = Config(reverse_sort=True)
    result_reverse = sorted_imports(parsed_content, config_reverse)
    assert result_reverse == "from itertools import chain\nfrom collections import defaultdict\nimport sys\nimport os\n\nx = 1\n"

    # Test with no imports
    parsed_content_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    result_no_imports = sorted_imports(parsed_content_no_imports, config)
    assert result_no_imports == "x = 1\n"

    # Test with place_imports
    parsed_content_place = parse.ParsedContent(
        lines_without_imports=["x = 1", "y = 2"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"y = 2": "THIRDPARTY"}
    )
    result_place = sorted_imports(parsed_content_place, config)
    assert result_place == "x = 1\ny = 2\nimport os\n"


# LLM-generated content at query #12
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
    config = Config()
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
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
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
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "from itertools import chain\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True, lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": [], "unused": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["unused"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
    )
    assert result == expected

    # Test case 6: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"django": []},
                "from": {"flask": ["Flask"]},
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
    expected = (
        "# Third Party Imports\n"
        "from flask import Flask\n"
        "\n"
        "import django\n"
        "\n"
    )
    assert result == expected

    # Test case 7: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "STDLIB": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "\n"
        "def main():\n"
        "    pass\n"
    )
    assert result == expected

    # Test case 8: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "def main():", "    pass"],
        imports={
            "STDLIB": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"STDLIB": ["import sys"]},
        import_placements={"# Placeholder": "STDLIB"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "# Placeholder\n"
        "import sys\n"
        "\n"
        "def main():\n"
        "    pass\n"
    )
    assert result == expected


# LLM-generated content at query #13
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["argv"], "json": ["load"]},
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
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import argv\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={
            "future": "Future Imports",
            "stdlib": "Standard Library",
            "thirdparty": "Third Party",
        }
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Future Imports\n"
        "from __future__ import annotations\n"
        "\n"
        "# Standard Library\n"
        "import os\n"
        "\n"
        "# Third Party\n"
        "import numpy\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["argv"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os", "json"])
    result = sorted_imports(parsed, config)
    expected = (
        "import zlib\n"
        "\n"
        "from sys import argv\n"
        "\n"
    )
    assert result == expected

    # Test case 6: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "import os\n"
        "import numpy\n"
        "\n"
    )
    assert result == expected

    # Test case 7: With force_sort_within_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["argv"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import argv\n"
        "\n"
    )
    assert result == expected

    # Test case 8: With lines_between_sections=2
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "\n"
        "import os\n"
        "\n"
    )
    assert result == expected

    # Test case 9: With lines_after_imports=2
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():\n    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import numpy\n"
        "\n"
        "\n"
        "def main():\n"
        "    pass\n"
    )
    assert result == expected

    # Test case 10: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    expected = "IMPORT NUMPY\n"
    assert result == expected


# LLM-generated content at query #14
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
    assert sorted_imports(parsed) == "\n"

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
    assert sorted_imports(parsed) == "print('hello')\n"

    # Test case 3: Basic sorting of imports
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
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 4: With custom config
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    assert "\n\n" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result
    assert "import os\nimport sys" in result

    # Test case 5: With forced separate sections
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        forced_separate=["os", "sys"],
    )
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 6: With star imports
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        star_first=True,
    )
    result = sorted_imports(parsed, config)
    assert "from collections import *" in result
    assert "from itertools import chain" in result

    # Test case 7: With custom import headings
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 8: With custom import footers
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_footers={"thirdparty": "End of Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result
    assert "# End of Third Party Imports" in result

    # Test case 9: With custom formatting function
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        formatting_function=lambda code, extension, config: code.upper(),
    )
    result = sorted_imports(parsed, config, extension="py")
    assert "IMPORT OS\nIMPORT SYS" in result
    assert "FROM COLLECTIONS IMPORT DEFAULTDICT" in result
    assert "FROM ITERTOOLS IMPORT CHAIN" in result

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os", "import sys"]},
        import_placements={"def main()": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "def main():" in result


# LLM-generated content at query #15
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
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
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": {"*"}, "collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from os import *\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import print_function\n\n\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import defaultdict\n\n\n\nx = 1\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
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
        import_placements={}
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"


# LLM-generated content at query #16
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports in parsed content
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    assert sorted_imports(parsed_no_imports) == "print('hello')\n"

    # Test case 2: Basic import sorting
    parsed_basic = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed_basic)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 3: With config modifications
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
    parsed_config = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed_config, config)
    assert result.startswith("\n\nfrom itertools import chain\nfrom collections import defaultdict\n\nimport sys\nimport os")


# LLM-generated content at query #17
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
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
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            },
            "FIRSTPARTY": {
                "straight": {"myapp": []},
                "from": {"myapp.utils": ["helper"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "import myapp" in result
    assert "from collections import defaultdict" in result
    assert "from myapp.utils import helper" in result

    # Test with from_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result.startswith("from collections import defaultdict\n\nimport os")

    # Test with star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["*"],
                    "os": ["path"]
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert "from collections import *" in result
    assert result.index("from collections import *") < result.index("from os import path")

    # Test with import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert "import os" not in result
    assert "import sys" in result

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1"

    # Test with place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1", "y = 2"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"y = 2": "THIRDPARTY"}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\ny = 2\nimport os\n"


# LLM-generated content at query #18
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
    expected = "\n".join([
        "import os",
        "import sys",
        "",
        "from collections import defaultdict",
        "",
        "print('hello')",
    ])
    assert result == expected

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {"sys": ["argv"]}},
            "THIRDPARTY": {"straight": {"django": ["django"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={
            "future": "Future imports",
            "stdlib": "Standard library imports",
            "thirdparty": "Third party imports",
        }
    )
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "# Future imports",
        "from __future__ import annotations",
        "",
        "# Standard library imports",
        "import os",
        "",
        "from sys import argv",
        "",
        "# Third party imports",
        "import django",
        "",
        "print('hello')",
    ])
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
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "import sys",
        "",
        "from collections import defaultdict",
        "",
        "print('hello')",
    ])
    assert result == expected

    # Test case 5: With force_sort_within_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"django": ["django"], "flask": ["flask"]},
                "from": {"numpy": ["array"], "pandas": ["DataFrame"]},
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
    expected = "\n".join([
        "import django",
        "import flask",
        "",
        "from numpy import array",
        "from pandas import DataFrame",
        "",
        "print('hello')",
    ])
    assert result == expected


# LLM-generated content at query #19
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

    # Test with forced_separate
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
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

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
    config = Config(remove_imports=["os", "sys"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\n"

    # Test with lines_between_types
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
    config = Config(lines_between_types=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n"

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
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

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

    # Test with import_footers
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
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n# End of Third Party Imports\n"

    # Test with lines_between_sections
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
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with ensure_newline_before_comments
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
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with formatting_function
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
    config = Config(formatting_function=lambda x, y, z: x)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with lines_before_imports
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
    config = Config(lines_before_imports=1)
    result = sorted_imports(parsed, config)
    assert result == "\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with lines_after_imports
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
    config = Config(lines_after_imports=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n"




# LLM-generated content at query #20
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
    assert result == "print('hello')"

    # Test case 3: Basic import sorting
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
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "import os\nimport zlib\n\nfrom json import load\nfrom sys import path"
    assert result == expected

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = "from __future__ import annotations\n\nimport os\nimport zlib"
    assert result == expected

    # Test case 5: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "from __future__ import annotations\n\nimport os\nimport zlib"
    assert result == expected

    # Test case 6: With custom import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = "# Third Party Imports\nimport os\nimport zlib"
    assert result == expected

    # Test case 7: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "from __future__ import annotations\n\n\n\nimport os\nimport zlib"
    assert result == expected

    # Test case 8: With star_first=True
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = "from sys import *\nfrom json import load"
    assert result == expected

    # Test case 9: With from_first=True
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
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = "from json import load\nfrom sys import path\n\nimport os\nimport zlib"
    assert result == expected

    # Test case 10: With force_sort_within_sections=True
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
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    expected = "import os\nimport zlib\n\nfrom json import load\nfrom sys import path"
    assert result == expected


# LLM-generated content at query #21
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed)
    assert result == "print('hello')\n"

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
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config=config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"future": "Future imports", "thirdparty": "Third party imports"},
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config=config)
    expected = (
        "# Future imports\n"
        "from __future__ import print_function\n"
        "\n"
        "# Third party imports\n"
        "import os\n"
        "import sys\n"
        "\n"
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
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config=config)
    expected = (
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
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config=config)
    expected = (
        "from __future__ import print_function\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #22
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
        sections=[],
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
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    )

    # Test case 3: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == (
        "import sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    )

    # Test case 4: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            },
        },
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import annotations\nimport os\nimport sys\n\n"
        "from collections import Counter, defaultdict\n\nprint('hello')"
    )

    # Test case 5: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["*"],
                    "os": ["path"],
                },
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from collections import *\nfrom os import path\n\nprint('hello')"
    )

    # Test case 6: With from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from collections import Counter, defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    )

    # Test case 7: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == (
        "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    )

    # Test case 8: With import_footers
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\n# End of Third Party Imports\nprint('hello')"
    )

    # Test case 9: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            },
        },
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import Counter, defaultdict\n\nprint('hello')"
    )

    # Test case 10: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\nimport sys\n\nfrom collections import Counter, defaultdict\n\n\nprint('hello')"
    )


# LLM-generated content at query #23
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
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 2: Single import
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": ["import os"]}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "import os\n\nprint('hello')\n"

    # Test case 3: Multiple imports in different sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["from __future__ import annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": ["import numpy"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=4,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "\n"
        "import numpy\n"
        "\n"
        "print('hello')\n"
    )
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["import numpy"], "pandas": ["import pandas"]},
                "from": {"numpy": ["from numpy import array"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=4,
        place_imports={},
        import_placements={},
    )
    config = Config(
        force_sort_within_sections=True,
        from_first=True,
        lines_between_sections=2,
    )
    expected = (
        "from numpy import array\n"
        "\n"
        "import numpy\n"
        "import pandas\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["import numpy"]},
                "from": {"numpy": ["from numpy import array"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["THIRDPARTY"])
    expected = (
        "from numpy import array\n"
        "import numpy\n"
        "\n"
        "print('hello')\n"
    )
    assert sorted_imports(parsed, config) == expected


# LLM-generated content at query #24
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["defaultdict", "Counter"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert "# Test file" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import Counter, defaultdict" in result

    # Test case 2: No sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "THIRDPARTY": {"straight": {"os": None}, "from": {"collections": ["Counter"]}}
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "import __future__" in result
    assert "import os" in result
    assert "from collections import Counter" in result

    # Test case 3: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {"django": None, "flask": None},
                "from": {"os": ["path"], "sys": ["argv"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert "import django" in result
    assert "import flask" in result
    assert "from os import path" in result
    assert "from sys import argv" in result

    # Test case 4: Remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None, "sys": None},
                "from": {"collections": ["Counter"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert "import os" not in result
    assert "import sys" in result

    # Test case 5: Custom import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": None},
                "from": {"pandas": ["DataFrame"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result
    assert "import numpy" in result
    assert "from pandas import DataFrame" in result

    # Test case 6: Lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": None}, "from": {}},
            "STDLIB": {"straight": {"os": None}, "from": {}},
            "THIRDPARTY": {"straight": {"django": None}, "from": {}}
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    future_idx = lines.index("import __future__")
    os_idx = lines.index("import os")
    django_idx = lines.index("import django")
    assert (os_idx - future_idx) > 2
    assert (django_idx - os_idx) > 2

    # Test case 7: From first config
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {"sys": ["argv"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    from_idx = lines.index("from sys import argv")
    import_idx = lines.index("import os")
    assert from_idx < import_idx

    # Test case 8: Star first config
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"numpy": ["*"], "pandas": ["DataFrame"]}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    star_idx = lines.index("from numpy import *")
    dataframe_idx = lines.index("from pandas import DataFrame")
    assert star_idx < dataframe_idx

    # Test case 9: Ensure newline before comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", "x = 1  # comment"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    import_idx = lines.index("import os")
    comment_idx = lines.index("x = 1  # comment")
    assert lines[comment_idx - 1] == ""
    assert comment_idx - import_idx == 2

    # Test case 10: Place imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", "x = 1", "# PLACE_HOLDER"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": None},
                "from": {}
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=3,
        place_imports={"PLACE_HOLDER": ["import sys"]},
        import_placements={"# PLACE_HOLDER": "PLACE_HOLDER"},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    assert lines.index("import sys") == lines.index("# PLACE_HOLDER") + 1


# LLM-generated content at query #25
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
    assert sorted_imports(parsed, config) == ""

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
    assert sorted_imports(parsed, config) == "print('hello')"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["import sys"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    expected = "import os\nimport zlib\n\nfrom sys import sys"
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {"sys": ["import sys"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    expected = "from sys import sys\n\nimport os\nimport zlib"
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    expected = "import os\n\nimport zlib"
    assert sorted_imports(parsed, config) == expected

    # Test case 6: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    expected = "import os\nimport zlib"
    assert sorted_imports(parsed, config) == expected

    # Test case 7: With star_first config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "sys": ["import *"],
                    "os": ["import path"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    expected = "from sys import *\nfrom os import path"
    assert sorted_imports(parsed, config) == expected

    # Test case 8: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    expected = "# Third Party Imports\nimport zlib"
    assert sorted_imports(parsed, config) == expected

    # Test case 9: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"], "os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    expected = "import zlib"
    assert sorted_imports(parsed, config) == expected

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "print('world')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": ["zlib"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import zlib"]},
        import_placements={"print('hello')": "THIRDPARTY"},
    )
    config = DEFAULT_CONFIG
    expected = "print('hello')\nimport zlib\n\nprint('world')"
    assert sorted_imports(parsed, config) == expected


# LLM-generated content at query #26
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 3: With custom config
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        import_headings={"THIRDPARTY": "Third Party Imports"}
    )
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n"

    # Test case 4: With forced separate sections
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            },
            "LOCALFOLDER": {
                "straight": {"sys": []},
                "from": {}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\nimport sys\n\n\nx = 1\n"

    # Test case 5: With no sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\nx = 1\n"


# LLM-generated content at query #27
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
    )
    config = DEFAULT_CONFIG
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    assert result == expected

    # Test case 3: With sections and headings
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
    )
    config = Config(import_headings={"future": "Future", "thirdparty": "Third Party"})
    result = sorted_imports(parsed, config)
    expected = "\n# Future\nfrom __future__ import annotations\n\n# Third Party\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    assert result == expected

    # Test case 4: With remove_imports
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
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = "\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
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
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import annotations\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    assert result == expected

    # Test case 6: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["from collections import *"],
                    "os": ["from os import path"],
                },
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import *\nfrom os import path\n\nprint('hello')"
    assert result == expected

    # Test case 7: With from_first
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
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"
    assert result == expected

    # Test case 8: With lines_between_sections
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
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    assert result == expected

    # Test case 9: With lines_after_imports
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
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\n\nprint('hello')"
    assert result == expected

    # Test case 10: With ensure_newline_before_comments
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
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"
    assert result == expected


# LLM-generated content at query #28
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
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["collections.OrderedDict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import OrderedDict\n\nprint('hello')"

    # Test case 3: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["django.conf"]}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\nimport os\n\nfrom django import conf\n\nprint('hello')"

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["collections.OrderedDict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    assert result == "from collections import OrderedDict\n\nimport os\nimport sys\n\nprint('hello')"

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["collections.OrderedDict"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nfrom collections import OrderedDict\n\nprint('hello')"


# LLM-generated content at query #29
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nx = 1"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import __future__\nimport os\n\nx = 1"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(forced_separate=["THIRDPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nx = 1"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nx = 1"

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
    assert result == "x = 1"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {"sys": ["sys"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import sys\n\nimport os\n\nx = 1"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "STDLIB": {
                "straight": {},
                "from": {"sys": ["*"], "os": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import *\nfrom os import path\n\nx = 1"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "# Standard Library\nimport os\n\nx = 1"

    # Test with import_footers
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_footers={"stdlib": "End Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n# End Standard Library\n\nx = 1"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import __future__\n\n\nimport os\n\nx = 1"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nx = 1"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\nx = 1"

    # Test with ensure_newline_before_comments
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nx = 1"

    # Test with formatting_function
    def custom_format(code, extension, config):
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    assert result == "import os\r\n\r\nx = 1"

    # Test with place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1", "y = 2"],
        imports={"STDLIB": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"STDLIB": ["import sys"]},
        import_placements={"x = 1": "STDLIB"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nx = 1\nimport sys\n\ny = 2"


# LLM-generated content at query #30
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            },
            "FIRSTPARTY": {
                "straight": {"my_module": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import my_module\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nprint('hello')"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": {"*"},
                    "module2": {"function1"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import function1\n\nprint('hello')"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nprint('hello')"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import print_function\n\n\nimport os\n\nprint('hello')"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nprint('hello')"


# LLM-generated content at query #31
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
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert result == expected

    # Test case 4: With config modifications
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import OrderedDict\n\nimport os\nimport sys\n"
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert result == expected

    # Test case 6: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\nimport os\nimport sys\nfrom collections import OrderedDict\n"
    assert result == expected

    # Test case 7: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    result = sorted_imports(parsed, config)
    expected = "\n# Third Party Imports\nimport os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert result == expected

    # Test case 8: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os", "import sys"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "# Placeholder\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 9: With formatting function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("import", "from").replace("from", "import")

    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    expected = "\nfrom os\nfrom sys\n\nimport collections import OrderedDict\n"
    assert result == expected

    # Test case 10: With lines_before_imports and lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_before_imports=2, lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = "\n\nimport os\nimport sys\n\nfrom collections import OrderedDict\n\n\nprint('hello')\n"
    assert result == expected


# LLM-generated content at query #32
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "import os\nimport sys\n\nfrom collections import defaultdict\n\nimport my_module\n\nprint('hello')"

    # Test with from_first=True
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed, config_from_first)
    assert result_from_first == "from collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')"

    # Test with star_first=True
    parsed_star = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["function"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star == "from module1 import *\nfrom module2 import function\n\nprint('hello')"

    # Test with custom import headings
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed, config_headings)
    assert result_headings == "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "print('hello')"

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    config_formatting = Config(formatting_function=custom_format)
    result_formatting = sorted_imports(parsed, config_formatting)
    assert result_formatting == "import os\r\nimport sys\r\n\r\nfrom collections import defaultdict\r\n\r\nprint('hello')"


# LLM-generated content at query #33
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        sections=[],
        imports={},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = DEFAULT_CONFIG
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
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        force_sort_within_sections=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(forced_separate=["LOCALFOLDER"])
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

    # Test case 6: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(remove_imports=["from collections import defaultdict"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 7: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY", "FUTURE"],
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
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 8: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.argv"]},
                "from": {
                    "collections": ["defaultdict"],
                    "typing": ["*"],  # Star import
                },
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from typing import *\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
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
    assert result == "from collections import defaultdict\nimport os\nimport sys\n"

    # Test case 2: No sections
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
    assert result == "from __future__ import absolute_import\nfrom collections import defaultdict\nimport os\nimport sys\n"

    # Test case 3: Force sort within sections
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
    assert result == "from collections import defaultdict\nimport os\nimport sys\n"

    # Test case 4: Star imports first
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

    # Test case 5: From imports first
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

    # Test case 6: Lines between sections
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
    assert result == "from __future__ import absolute_import\n\n\nfrom collections import defaultdict\nimport os\nimport sys\n"

    # Test case 7: Import headings
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
    assert result == "# Third Party Imports\nfrom collections import defaultdict\nimport os\nimport sys\n"

    # Test case 8: Import footers
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
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport os\nimport sys\n\n# End of Third Party Imports\n"

    # Test case 9: Lines before imports
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
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom collections import defaultdict\nimport os\nimport sys\n"

    # Test case 10: Lines after imports
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
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport os\nimport sys\n\n\n"


# LLM-generated content at query #35
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 3: With section headings
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\nimport sys\n\nx = 1\n"

    # Test case 4: With forced separate sections
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
            "LOCALFOLDER": {
                "straight": {"sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nimport sys\n\nx = 1\n"

    # Test case 5: With remove_imports
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nx = 1\n"

    # Test case 6: With lines_between_sections
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\n\nimport sys\n\nx = 1\n"

    # Test case 7: With lines_after_imports
    config = Config(lines_after_imports=2)
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
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nx = 1\n"

    # Test case 8: With no_sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"json": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "import json\nimport os\nimport sys\n\nx = 1\n"

    # Test case 9: With from_first
    config = Config(from_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "from sys import path\n\nimport os\n\nx = 1\n"

    # Test case 10: With star_first
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "from os import *\nfrom sys import path\n\nx = 1\n"


# LLM-generated content at query #36
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 3: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(
        from_first=True,
        lines_between_sections=2,
        lines_between_types=1,
    )
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == (
        "from collections import defaultdict\n"
        "\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 5: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["*"],
                    "collections": ["defaultdict"],
                },
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from os import *\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 6: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import print_function\n"
        "\n"
        "from collections import defaultdict\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 7: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == (
        "# Third Party Imports\n"
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 8: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "\n"
        "\n"
        "print('hello')\n"
    )

    # Test case 9: With formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
            }
        },
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    result = sorted_imports(parsed, config)
    assert result == (
        "IMPORT OS\n"
        "\n"
        "PRINT('HELLO')\n"
    )

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "print('hello')"],
        import_index=0,
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
            }
        },
        line_separator="\n",
        original_line_count=2,
        place_imports={"THIRDPARTY": ["import sys"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "\n"
        "# Placeholder\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )


# LLM-generated content at query #37
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        sections=[],
        imports={},
        place_imports={},
        import_placements={},
        original_line_count=1,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import OrderedDict\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        sections=["FUTURE", "THIRDPARTY"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            },
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import __future__\n"
        "import os\n"
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #38
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict, OrderedDict\n"
    )
    assert result == expected

    # Test case 4: With config modifications
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 5: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        forced_separate=["LOCALFOLDER"],
        imports={"LOCALFOLDER": {"straight": {"local": ["local"]}}},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import local\n"
    )
    assert result == expected


# LLM-generated content at query #39
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"django": []}, "from": {"numpy": ["array"]}}
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(import_headings={"future": "Future", "stdlib": "Standard Library", "thirdparty": "Third Party"})
    result = sorted_imports(parsed, config)
    assert "# Future" in result
    assert "# Standard Library" in result
    assert "# Third Party" in result

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": [], "unused": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=["unused"])
    result = sorted_imports(parsed, config)
    assert "unused" not in result

    # Test case 5: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"django": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result.count("\n\n") >= 2

    # Test case 6: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"django": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "# Future" in result
    assert "# Standard Library" not in result
    assert "# Third Party" not in result

    # Test case 7: With reverse_sort
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    assert result.startswith("import sys")

    # Test case 8: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"numpy": ["*", "array"], "collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result.index("from numpy import *") < result.index("from numpy import array")

    # Test case 9: With from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result.index("from collections") < result.index("import os")

    # Test case 10: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config, extension="py")
    assert "\r\n" in result


# LLM-generated content at query #40
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": [], "sys": []}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

    # Test case 2: No sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\n"

    # Test case 3: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {"os": [], "sys": []},
                "from": {"os.path": ["join"], "sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom os.path import join\nfrom sys import argv\n"

    # Test case 4: Custom sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "CUSTOM": {"straight": {"custom": []}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["CUSTOM"])
    result = sorted_imports(parsed, config)
    assert result == "import custom\n\nimport os\n"

    # Test case 5: Empty imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == ""


# LLM-generated content at query #41
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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
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
        place_imports={},
        import_placements={},
    )
    expected = "import os\nimport sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed) == expected

    # Test case 4: With config options
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
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
        place_imports={},
        import_placements={},
    )
    expected = "from collections import defaultdict\n\nimport os\nimport sys\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With remove_imports
    config = Config(
        remove_imports=["os"],
    )
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
        place_imports={},
        import_placements={},
    )
    expected = "import sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With import headings
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
    )
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
        place_imports={},
        import_placements={},
    )
    expected = "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With star_first
    config = Config(
        star_first=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["*"],
                    "os": ["path"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "from collections import *\nfrom os import path\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 8: With no_sections
    config = Config(
        no_sections=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "from __future__ import annotations\nimport os\nimport sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 9: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
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
        import_placements={"print('hello')": "THIRDPARTY"},
    )
    expected = "import os\nprint('hello')\nimport sys\n"
    assert sorted_imports(parsed) == expected

    # Test case 10: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    config = Config(
        formatting_function=custom_format,
    )
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    expected = "import os\r\n"
    assert sorted_imports(parsed, config=config) == expected


# LLM-generated content at query #42
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting of imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict", "OrderedDict"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict", "OrderedDict"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "from __future__ import print_function\n\nfrom collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with from_first=True
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict", "OrderedDict"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first == "from collections import OrderedDict, defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test with star_first=True
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {
                    "collections": {"defaultdict", "OrderedDict"},
                    "numpy": {"*"}
                }
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first == "from numpy import *\nfrom collections import OrderedDict, defaultdict\n\nimport os\n\nx = 1\n"

    # Test with custom import headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings == "# Third Party Imports\nfrom collections import defaultdict\n\nimport os\n\nx = 1\n"

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
    config_no_imports = Config()
    result_no_imports = sorted_imports(parsed_no_imports, config_no_imports)
    assert result_no_imports == "x = 1\n"


# LLM-generated content at query #43
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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "import os\nimport sys\n\nfrom collections import OrderedDict"
    assert sorted_imports(parsed) == expected

    # Test case 4: With config options
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "from collections import OrderedDict\n\nimport os\nimport sys"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With import headings
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
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
    )
    expected = "# Third Party Imports\nimport os"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With forced separate sections
    config = Config(
        forced_separate=["FUTURE", "STDLIB"],
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "import __future__\n\nimport os"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With star_first option
    config = Config(
        star_first=True,
    )
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
        place_imports={},
        import_placements={},
    )
    expected = "from module1 import *\nfrom module2 import function1"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 8: With lines_after_imports
    config = Config(
        lines_after_imports=2,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def func():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "import os\n\ndef func():"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 9: With remove_imports
    config = Config(
        remove_imports=["import os"],
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
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
    )
    expected = "import sys"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 10: With no_sections
    config = Config(
        no_sections=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "import __future__\nimport os"
    assert sorted_imports(parsed, config=config) == expected


# LLM-generated content at query #44
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nimport numpy\nimport pandas\n\nfrom collections import defaultdict\n"
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport numpy\nimport pandas\n"
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {}
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = "\nimport numpy\n\nimport my_module\n"
    assert result == expected


# LLM-generated content at query #45
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
    config = Config()
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 3: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "import os",
        "import sys",
        "",
        "from collections import defaultdict",
        "from itertools import chain",
    ])
    assert result == expected

    # Test case 4: With custom config (reverse_sort=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "import sys",
        "import os",
        "",
        "from itertools import chain",
        "from collections import defaultdict",
    ])
    assert result == expected

    # Test case 5: With star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "from": {
                    "os": ["*"], "sys": ["path"], "collections": ["defaultdict"]
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "from os import *",
        "from collections import defaultdict",
        "from sys import path",
    ])
    assert result == expected

    # Test case 6: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            },
            "FIRSTPARTY": {"straight": {"my_module": ["my_module"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["FUTURE", "THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "from __future__ import annotations",
        "",
        "from collections import defaultdict",
        "import my_module",
        "import os",
        "import sys",
    ])
    assert result == expected

    # Test case 7: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
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
    expected = "\n".join([
        "# Third Party Imports",
        "import os",
    ])
    assert result == expected

    # Test case 8: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}},
            "FIRSTPARTY": {"straight": {"my_module": ["my_module"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "import os",
        "",
        "",
        "import my_module",
    ])
    assert result == expected

    # Test case 9: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = "\n".join([
        "import os",
        "",
        "",
        "def main():",
        "    pass",
    ])
    assert result == expected

    # Test case 10: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = "import sys"
    assert result == expected


# LLM-generated content at query #46
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nos\nsys\n\nfrom collections import Counter, defaultdict\n\nx = 1\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "Counter"]},
            },
            "FIRSTPARTY": {
                "straight": {"myapp": ["myapp"]},
                "from": {"myapp.utils": ["helper"]},
            },
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "\nmyapp\nos\nsys\n\nfrom collections import Counter, defaultdict\nfrom myapp.utils import helper\n\nx = 1\n"

    # Test with star_first=True
    parsed_star = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {
                    "collections": ["defaultdict"],
                    "numpy": ["*"],
                },
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star == "\nos\n\nfrom numpy import *\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with custom import headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_headings = Config(import_headings={"thirdparty": "Third-party imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert result_headings == "\n# Third-party imports\nos\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result_no_imports = sorted_imports(parsed_no_imports)
    assert result_no_imports == "x = 1\n"


# LLM-generated content at query #47
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
    )
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import OrderedDict" in result

    # Test case 4: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"future": "Future", "stdlib": "Standard Library"},
        dedup_headings=True,
    )
    result = sorted_imports(parsed, config)
    assert "# Future" in result
    assert "# Standard Library" in result
    assert "from __future__ import annotations" in result
    assert "import os" in result

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"django": ["django"]}, "from": {}},
            "FIRSTPARTY": {"straight": {"myapp": ["myapp"]}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert "import django" in result
    assert "import myapp" in result

    # Test case 6: With no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["os"]}, "from": {}},
            "FIRSTPARTY": {"straight": {"sys": ["sys"]}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result

    # Test case 7: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["function"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert "from module1 import *" in result
    assert "from module2 import function" in result
    assert result.index("from module1 import *") < result.index("from module2 import function")

    # Test case 8: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "# IMPORTS"],
        imports={"THIRDPARTY": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"# IMPORTS": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert result.index("import os") > result.index("# IMPORTS")

    # Test case 9: With formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    result = sorted_imports(parsed, config)
    assert "IMPORT OS" in result

    # Test case 10: With ensure_newline_before_comments
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "# comment"],
        imports={"THIRDPARTY": {"straight": {"os": ["os"]}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert "\n# comment" in result


# LLM-generated content at query #48
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
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = """from collections import defaultdict

import os
import sys

print('hello')"""
    assert result == expected

    # Test case 3: With sections and headings
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        lines_between_sections=1,
    )
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
    result = sorted_imports(parsed, config)
    expected = """# Third Party Imports
from collections import defaultdict

import os
import sys

print('hello')"""
    assert result == expected

    # Test case 4: With remove_imports
    config = Config(remove_imports=["os"])
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
    result = sorted_imports(parsed, config)
    expected = """from collections import defaultdict

import sys

print('hello')"""
    assert result == expected

    # Test case 5: With no_sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}},
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config)
    expected = """from __future__ import __future__

from collections import defaultdict

import os
import sys

print('hello')"""
    assert result == expected


# LLM-generated content at query #49
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import numpy" in result
    assert "import pandas" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "x = 1\n"

    # Test with forced_separate
    config_forced = Config(forced_separate=["TEST"])
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "TEST": {
                "straight": {"pytest": []},
                "from": {"pytest": ["fixture"]},
            },
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "import pytest" in result_forced
    assert "from pytest import fixture" in result_forced
    assert "import numpy" in result_forced
    assert "from collections import defaultdict" in result_forced

    # Test with no_sections
    config_no_sections = Config(no_sections=True)
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "from __future__ import annotations" in result_no_sections
    assert "import numpy" in result_no_sections
    assert "from collections import defaultdict" in result_no_sections

    # Test with only_sections
    config_only_sections = Config(only_sections=True)
    parsed_only_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_only_sections = sorted_imports(parsed_only_sections, config_only_sections)
    assert "import numpy" in result_only_sections
    assert "import pandas" in result_only_sections
    assert "from collections import defaultdict" in result_only_sections
    assert "from itertools import chain" in result_only_sections

    # Test with star_first
    config_star_first = Config(star_first=True)
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module": ["*"], "other": ["function"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert "from module import *" in result_star_first
    assert "from other import function" in result_star_first
    assert result_star_first.index("from module import *") < result_star_first.index("from other import function")

    # Test with from_first
    config_from_first = Config(from_first=True)
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from collections import defaultdict") < result_from_first.index("import numpy")

    # Test with import_headings
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings

    # Test with import_footers
    config_footers = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    parsed_footers = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_footers = sorted_imports(parsed_footers, config_footers)
    assert "# End of Third Party Imports" in result_footers

    # Test with lines_between_sections
    config_lines_between = Config(lines_between_sections=2)
    parsed_lines_between = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_lines_between = sorted_imports(parsed_lines_between, config_lines_between)
    assert result_lines_between.count("\n\n") >= 2

    # Test with lines_between_types
    config_lines_between_types = Config(lines_between_types=1)
    parsed_lines_between_types = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_lines_between_types = sorted_imports(parsed_lines_between_types, config_lines_between_types)
    assert result_lines_between_types.index("import numpy") < result_lines_between_types.index("")
    assert result_lines_between_types.index("") < result_lines_between_types.index("from collections import defaultdict")

    # Test with reverse_sort
    config_reverse = Config(reverse_sort=True)
    parsed_reverse = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result_reverse = sorted_imports(parsed_reverse, config_reverse)
    assert result_reverse.index("import pandas") < result_reverse.index("import numpy")
    assert result_reverse.index("from itertools import chain") < result_reverse.index("from collections import defaultdict")

    # Test with remove_imports
    config_remove = Config(remove_imports=["numpy"])
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": [], "pandas


# LLM-generated content at query #50
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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"django": [], "flask": []},
                "from": {"os": ["path"], "sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "import django\n"
        "import flask\n"
        "\n"
        "from os import path\n"
        "from sys import argv\n"
    )
    assert sorted_imports(parsed) == expected

    # Test case 4: With config modifications
    config = Config(
        force_sort_within_sections=True,
        lines_between_sections=2,
        lines_between_types=1,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"django": [], "flask": []},
                "from": {"os": ["path"], "sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "import django\n"
        "import flask\n"
        "\n"
        "from os import path\n"
        "\n"
        "from sys import argv\n"
    )
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With forced separate sections
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"django": []}, "from": {}},
            "LOCALFOLDER": {"straight": {"my_module": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "import django\n"
        "\n"
        "import my_module\n"
    )
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With import headings
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        dedup_headings=True,
    )
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"django": []},
                "from": {"os": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "# Third Party Imports\n"
        "import django\n"
        "\n"
        "from os import path\n"
    )
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With star_first enabled
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "from os import *\n"
        "from sys import argv\n"
    )
    assert sorted_imports(parsed, config=config) == expected

    # Test case 8: With remove_imports
    config = Config(remove_imports=["import os"])
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    expected = ""
    assert sorted_imports(parsed, config=config) == expected

    # Test case 9: With lines_before_imports and lines_after_imports
    config = Config(lines_before_imports=2, lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = (
        "\n"
        "\n"
        "import os\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert sorted_imports(parsed, config=config) == expected

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Place imports here", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"# Place imports here": ["import os"]},
        import_placements={"# Place imports here": "# Place imports here"},
    )
    expected = (
        "# Place imports here\n"
        "import os\n"
        "\n"
        "print('hello')\n"
    )
    assert sorted_imports(parsed) == expected


# LLM-generated content at query #51
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nos\nsys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    assert sorted_imports(parsed_no_imports, config) == "print('hello')"

    # Test with forced_separate
    config_with_separate = Config(forced_separate=["numpy"])
    parsed_with_separate = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            },
            "numpy": {
                "straight": {"numpy": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    result = sorted_imports(parsed_with_separate, config_with_separate)
    assert "numpy" in result
    assert result.index("numpy") > result.index("sys")

    # Test with remove_imports
    config_with_remove = Config(remove_imports=["sys"])
    parsed_with_remove = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    result = sorted_imports(parsed_with_remove, config_with_remove)
    assert "sys" not in result
    assert "os" in result

    # Test with lines_between_sections
    config_with_lines = Config(lines_between_sections=2)
    parsed_with_lines = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    result = sorted_imports(parsed_with_lines, config_with_lines)
    assert result.count("\n\n") >= 2

    # Test with import_headings
    config_with_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed_with_headings = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    result = sorted_imports(parsed_with_headings, config_with_headings)
    assert "# Third Party Imports" in result


# LLM-generated content at query #52
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports in parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 3: With custom config (reverse_sort=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport sys\nimport os\n\nprint('hello')\n"
    assert result == expected

    # Test case 4: With forced_separate config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(forced_separate=["os"])
    result = sorted_imports(parsed, config)
    expected = "\nimport os\n\nfrom collections import defaultdict\n\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 5: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import annotations\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 6: With star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["*"],
                    "module2": ["func1"],
                    "module3": ["*"],
                },
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = "\nfrom module1 import *\nfrom module3 import *\nfrom module2 import func1\n\nprint('hello')\n"
    assert result == expected

    # Test case 7: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = "\n# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 8: With lines_between_sections=2
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import annotations\n\n\n\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nprint('hello')\n"
    assert result == expected

    # Test case 9: With lines_after_imports=2
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport os\nimport sys\n\n\n\nprint('hello')\n"
    assert result == expected

    # Test case 10: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = "\nfrom collections import defaultdict\n\nimport sys\n\nprint('hello')\n"
    assert result == expected


# LLM-generated content at query #53
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "sys\nzlib\n\nfrom os import path\n\nx = 1"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"zlib": [], "sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "__future__\nsys\nzlib\n\nfrom os import path\n\nx = 1"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "sys\nzlib\n\nfrom os import path\n\nx = 1"

    # Test with remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(remove_imports=["zlib"])
    result = sorted_imports(parsed, config)
    assert result == "sys\n\nfrom os import path\n\nx = 1"

    # Test with star first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from os import *\nfrom sys import path\n\nx = 1"

    # Test with from first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from os import path\n\nsys\n\nx = 1"

    # Test with import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nsys\n\nfrom os import path\n\nx = 1"

    # Test with lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "__future__\n\n\nsys\n\nfrom os import path\n\nx = 1"

    # Test with lines after imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "sys\n\nfrom os import path\n\n\n\nx = 1"

    # Test with formatting function
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": []},
                "from": {"os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(formatting_function=lambda x, y, z: x)
    result = sorted_imports(parsed, config)
    assert result == "sys\n\nfrom os import path\n\nx = 1"


# LLM-generated content at query #54
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
    result = sorted_imports(parsed)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports with default config
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
    result = sorted_imports(parsed)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With custom config (reverse_sort=True)
    config = Config(reverse_sort=True)
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
    result = sorted_imports(parsed, config=config)
    expected = (
        "import sys\n"
        "import os\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With forced_separate config
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {},
            },
            "LOCALFOLDER": {
                "straight": {"my_module": ["my_func"]},
                "from": {},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config=config)
    expected = (
        "import os\n"
        "\n"
        "import my_module\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With no_sections config
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    result = sorted_imports(parsed, config=config)
    expected = (
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #55
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
    assert sorted_imports(parsed) == "\n"

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
    assert sorted_imports(parsed) == "print('hello')\n"

    # Test case 3: Basic import sorting
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
    expected = "import os\nimport sys\nfrom collections import OrderedDict\n\n"
    assert sorted_imports(parsed) == expected

    # Test case 4: With config modifications
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
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
    expected = "from collections import OrderedDict\n\nimport sys\nimport os\n\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With forced_separate and no_sections
    config = Config(
        forced_separate=["os"],
        no_sections=True,
    )
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
    expected = "import os\nimport sys\nfrom collections import OrderedDict\n\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With import headings and footers
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
        import_footers={"thirdparty": "End of Third Party Imports"},
    )
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
    expected = "# Third Party Imports\nimport os\nimport sys\nfrom collections import OrderedDict\n\n# End of Third Party Imports\n\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With star_first enabled
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": ["*"], "sys": ["path"], "collections": ["OrderedDict"]
                },
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    expected = "from os import *\nfrom collections import OrderedDict\nfrom sys import path\n\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 8: With remove_imports
    config = Config(remove_imports=["os"])
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
    expected = "import sys\nfrom collections import OrderedDict\n\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 9: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    config = Config(formatting_function=custom_format)
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
        line_separator="\r\n",
        place_imports={},
        import_placements={},
    )
    expected = "import os\r\nimport sys\r\nfrom collections import OrderedDict\r\n\r\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os", "import sys"]},
        import_placements={"print('hello')": "THIRDPARTY"},
    )
    expected = "import os\nimport sys\nprint('hello')\n"
    assert sorted_imports(parsed) == expected


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + devstral-2512 t=0.8)      #
####################################################################


# LLM-generated content at query #1
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
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

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
        place_imports={},
        import_placements={},
        original_line_count=2,
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

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {"sys": ["argv"]}},
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library imports"}
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Future imports\n"
        "from __future__ import annotations\n"
        "\n"
        "# Standard library imports\n"
        "import os\n"
        "from sys import argv\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "pytest": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(remove_imports=["pytest"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {"sys": ["argv"]}},
        },
        place_imports={},
        import_placements={},
        original_line_count=2,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "import os\n"
        "from sys import argv\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=False)
    expected = "import os\nimport sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With sections and comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Main code"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}},
            "STDLIB": {"straight": {"os": ["os"], "sys": ["sys"]}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"future": "Future", "stdlib": "Standard Library"},
        lines_between_sections=1,
    )
    expected = "from __future__ import annotations\n\n# Future\n\nimport os\nimport sys\n\n# Standard Library\n\n# Main code\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    expected = "import sys\n\nfrom collections import defaultdict\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 6: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}},
            "STDLIB": {"straight": {"os": ["os"]}},
            "THIRDPARTY": {"straight": {"django": ["django"]}},
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    expected = "from __future__ import annotations\nimport django\nimport os\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 7: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "from": {
                    "numpy": ["*"],
                    "pandas": ["DataFrame"],
                }
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    expected = "from numpy import *\nfrom pandas import DataFrame\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 8: With formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=lambda x, y, z: x.upper())
    expected = "IMPORT OS\n"
    assert sorted_imports(parsed, config) == expected


# LLM-generated content at query #3
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 3: With custom config
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test case 4: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["os"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n"

    # Test case 5: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nfrom collections import defaultdict\n\nimport os\nimport sys\n\nx = 1\n"


# LLM-generated content at query #4
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
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
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "x = 1\n"

    # Test with forced separate sections
    config_with_separate = Config(forced_separate=["LOCALFOLDER"])
    parsed_with_separate = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "LOCALFOLDER": {
                "straight": {"local": set()},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_with_separate = sorted_imports(parsed_with_separate, config_with_separate)
    assert "import local\n" in result_with_separate
    assert "import os\n" in result_with_separate

    # Test with no sections
    config_no_sections = Config(no_sections=True)
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "from __future__ import print_function\n" in result_no_sections
    assert "import os\n" in result_no_sections
    assert "from sys import exit\n" in result_no_sections

    # Test with remove imports
    config_remove = Config(remove_imports=["from sys import *"])
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"*"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "from sys import *" not in result_remove
    assert "import os\n" in result_remove

    # Test with star_first
    config_star_first = Config(star_first=True)
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "sys": {"exit"},
                    "os": {"*"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first.index("from os import *") < result_star_first.index("from sys import exit")

    # Test with from_first
    config_from_first = Config(from_first=True)
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from sys import exit") < result_from_first.index("import os")

    # Test with import headings
    config_with_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed_with_headings = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_with_headings = sorted_imports(parsed_with_headings, config_with_headings)
    assert "# Third Party Imports" in result_with_headings
    assert result_with_headings.index("# Third Party Imports") < result_with_headings.index("import os")

    # Test with lines_between_sections
    config_lines_between = Config(lines_between_sections=2)
    parsed_lines_between = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_lines_between = sorted_imports(parsed_lines_between, config_lines_between)
    future_index = result_lines_between.index("from __future__ import print_function")
    os_index = result_lines_between.index("import os")
    assert os_index - future_index == 3  # 2 empty lines + 1 line with import

    # Test with lines_after_imports
    config_lines_after = Config(lines_after_imports=2)
    parsed_lines_after = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_lines_after = sorted_imports(parsed_lines_after, config_lines_after)
    os_index = result_lines_after.index("import os")
    x_index = result_lines_after.index("x = 1")
    assert x_index - os_index == 3  # 2 empty lines + 1 line with code


# LLM-generated content at query #5
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
    assert result == "print('hello')"

    # Test case 3: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict", "defaultdict"]},
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
        "from collections import OrderedDict, defaultdict\n"
        "\n"
        "import os, sys\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 4: With config modifications
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict", "defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict, defaultdict\n"
        "\n"
        "\n"
        "import sys, os\n"
        "\n"
        "\n"
        "print('hello')"
    )
    assert result == expected

    # Test case 5: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["OrderedDict"]},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import OrderedDict\n"
        "\n"
        "import os\n"
        "\n"
        "\n"
        "import sys\n"
        "\n"
        "print('hello')"
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys"]},
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
                "straight": {"os": ["os.path"], "sys": ["sys"]},
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
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys"]},
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
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys"]},
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
        remove_imports=["os"],
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import sys\n"
        "\n"
        "from collections import defaultdict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #7
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
    config = Config()
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(
        from_first=True,
        lines_between_sections=2,
        lines_between_types=1,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #8
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
    config = Config()
    assert sorted_imports(parsed, config) == ""

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
    config = Config()
    assert sorted_imports(parsed, config) == "print('hello')"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config()
    expected = "import os\nimport zlib\nfrom sys import path\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(reverse_sort=True)
    expected = "import zlib\nimport os\nfrom sys import path\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    expected = "import zlib\n\nimport os\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 6: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    expected = "# Third Party Imports\nimport zlib\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 7: With lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(lines_between_sections=2)
    expected = "import zlib\n\n\nimport os\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 8: With star_first option
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"sys": ["*"], "os": ["path"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(star_first=True)
    expected = "from sys import *\nfrom os import path\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 9: With no_sections option
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(no_sections=True)
    expected = "import os\nimport zlib\n"
    assert sorted_imports(parsed, config) == expected

    # Test case 10: With formatting function
    parsed = parse.ParsedContent(
        lines_without_imports=[],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(formatting_function=lambda code, ext, cfg: code.upper())
    expected = "IMPORT ZLIB\n"
    assert sorted_imports(parsed, config) == expected


# LLM-generated content at query #9
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={"STDLIB": {"straight": {"os": [], "sys": []}, "from": {}}},
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n"

    # Test with sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\nimport os\nimport sys\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport os\nimport sys\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {"os": []},
                "from": {"sys": [("path", "sys.path")]},
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
    assert result == "from sys import path\n\nimport os\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {},
                "from": {
                    "os": [("*", None)],
                    "sys": [("path", "sys.path")],
                },
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
    assert result == "from os import *\nfrom sys import path\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {"os": [], "sys": []},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"


# LLM-generated content at query #10
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
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
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
    expected = "\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 3: With custom config (force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.platform"]},
                "from": {"sys": ["sys.argv"], "os": ["os.getcwd"]},
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
    expected = "\nfrom os import os.getcwd\nfrom sys import sys.argv\nimport os, sys\n\nprint('hello')"
    assert result == expected

    # Test case 4: With remove_imports config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"], "sys": ["sys.platform"]},
                "from": {"sys": ["sys.argv"], "os": ["os.getcwd"]},
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
    expected = "\nimport sys\n\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 5: With import_headings config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    expected = "\n# Third Party Imports\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 6: With lines_between_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"from __future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = "\nfrom __future__ import print_function\n\n\n\nimport os\n\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 7: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "# PLACE_IMPORTS_HERE"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={"PLACE_IMPORTS_HERE": ["import os", "from sys import sys.argv"]},
        import_placements={"# PLACE_IMPORTS_HERE": "PLACE_IMPORTS_HERE"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = "print('hello')\n# PLACE_IMPORTS_HERE\nimport os\nfrom sys import sys.argv\n"
    assert result == expected

    # Test case 8: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"from __future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": ["my_function"]},
                "from": {},
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
    expected = "\nfrom __future__ import print_function\nimport my_module, os\n\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 9: With star_first config
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["sys.argv"]},
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
    expected = "\nfrom os import *\nfrom sys import sys.argv\n\nprint('hello')"
    assert result == expected

    # Test case 10: With formatting_function config
    def custom_formatter(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os.path"]},
                "from": {"sys": ["sys.argv"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(formatting_function=custom_formatter)
    result = sorted_imports(parsed, config)
    expected = "\r\nimport os\r\n\r\nfrom sys import sys.argv\r\n\r\nprint('hello')"
    assert result == expected


# LLM-generated content at query #11
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Empty parsed content
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        sections=[],
        import_index=-1,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "\n"

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        sections=[],
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
                "straight": {"numpy": ["numpy"], "pandas": ["pandas"]},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "import numpy\nimport pandas\n\n"

    # Test case 4: Multiple sections with imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "STDLIB": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"]},
                "from": {"pandas": ["DataFrame"]},
            },
        },
        sections=["FUTURE", "STDLIB", "THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import absolute_import\n\n"
        "import os\nimport sys\n\n"
        "import numpy\n\n"
        "from pandas import DataFrame\n\n"
    )

    # Test case 5: With custom config (reverse sort)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"], "pandas": ["pandas"]},
                "from": {},
            }
        },
        sections=["THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    assert result == "import pandas\nimport numpy\n\n"

    # Test case 6: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": ["my_module"]},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    assert result == "import numpy\n\nimport my_module\n\n"

    # Test case 7: With no sections config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["__future__"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"]},
                "from": {"pandas": ["DataFrame"]},
            },
        },
        sections=["FUTURE", "THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "from __future__ import absolute_import\n\n"
        "import numpy\n\n"
        "from pandas import DataFrame\n\n"
    )

    # Test case 8: With custom import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"]},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport numpy\n\n"

    # Test case 9: With lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "STDLIB": {
                "straight": {"os": ["os"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"]},
                "from": {},
            },
        },
        sections=["STDLIB", "THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\n\nimport numpy\n\n"

    # Test case 10: With remove_imports config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"numpy": ["numpy"], "pandas": ["pandas"]},
                "from": {},
            },
        },
        sections=["THIRDPARTY"],
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["pandas"])
    result = sorted_imports(parsed, config)
    assert result == "import numpy\n\n"


# LLM-generated content at query #12
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
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
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With custom config
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from json import load\n"
        "from sys import path\n"
        "\n"
        "import os\n"
        "import zlib\n"
        "\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {"json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        forced_separate=["FIRSTPARTY"],
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import zlib\n"
        "\n"
        "from sys import path\n"
        "\n"
        "\n"
        "import os\n"
        "\n"
        "from json import load\n"
        "\n"
    )
    assert result == expected

    # Test case 6: With star imports first
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from sys import *\n"
        "from json import load\n"
        "\n"
    )
    assert result == expected

    # Test case 7: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import zlib\n"
        "\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected

    # Test case 8: With no sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {"json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected


# LLM-generated content at query #13
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
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "print('hello')\n"

    # Test case 2: Single import
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={"THIRDPARTY": {"straight": {"os": ["import os"]}, "from": {}}},
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "import os\n\nprint('hello')\n"

    # Test case 3: Multiple imports, same section
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": ["import sys"], "os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\nprint('hello')\n"

    # Test case 4: Multiple sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["from __future__ import annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": ["import os"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "from __future__ import annotations\n\nimport os\n\nprint('hello')\n"

    # Test case 5: With config modifications
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": ["import sys"], "os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(lines_between_sections=2)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\n\nprint('hello')\n"

    # Test case 6: With from imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["from os import path"]},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = DEFAULT_CONFIG
    assert sorted_imports(parsed, config) == "from os import path\n\nprint('hello')\n"

    # Test case 7: With star imports and star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "os": ["from os import *"],
                    "sys": ["from sys import path"],
                },
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(star_first=True)
    assert sorted_imports(parsed, config) == "from os import *\nfrom sys import path\n\nprint('hello')\n"

    # Test case 8: With no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": ["from __future__ import annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": ["import os"]}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(no_sections=True)
    assert sorted_imports(parsed, config) == "from __future__ import annotations\nimport os\n\nprint('hello')\n"

    # Test case 9: With force_sort_within_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"sys": ["import sys"], "os": ["import os"]},
                "from": {},
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(force_sort_within_sections=True)
    assert sorted_imports(parsed, config) == "import os\nimport sys\n\nprint('hello')\n"

    # Test case 10: With import headings
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
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    assert sorted_imports(parsed, config) == "# Third Party Imports\nimport os\n\nprint('hello')\n"


# LLM-generated content at query #14
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
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections == "\nfrom __future__ import absolute_import\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with only_sections=True
    parsed_only_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_only_sections = Config(only_sections=True)
    result_only_sections = sorted_imports(parsed_only_sections, config_only_sections)
    assert result_only_sections == "\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with star_first=True
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first == "\nfrom module1 import *\nfrom module2 import func\n\nx = 1\n"

    # Test with from_first=True
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first == "\nfrom collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test with lines_between_types
    parsed_lines_between = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_lines_between = Config(lines_between_types=2)
    result_lines_between = sorted_imports(parsed_lines_between, config_lines_between)
    assert result_lines_between == "\nimport os\n\n\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with lines_between_sections
    parsed_lines_between_sections = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_lines_between_sections = Config(lines_between_sections=2)
    result_lines_between_sections = sorted_imports(
        parsed_lines_between_sections, config_lines_between_sections
    )
    assert result_lines_between_sections == "\nfrom __future__ import absolute_import\n\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with import_headings
    parsed_import_headings = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_import_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_import_headings = sorted_imports(parsed_import_headings, config_import_headings)
    assert result_import_headings == "\n# Third Party Imports\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with import_footers
    parsed_import_footers = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_import_footers = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result_import_footers = sorted_imports(parsed_import_footers, config_import_footers)
    assert result_import_footers == "\nimport os\n\nfrom collections import defaultdict\n\n# End of Third Party Imports\n\nx = 1\n"

    # Test with lines_before_imports
    parsed_lines_before = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_lines_before = Config(lines_before_imports=2)
    result_lines_before = sorted_imports(parsed_lines_before, config_lines_before)
    assert result_lines_before == "\n\nimport os\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with lines_after_imports
    parsed_lines_after = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_lines_after = Config(lines_after_imports=2)
    result_lines_after = sorted_imports(parsed_lines_after, config_lines_after)
    assert result_lines_after == "\nimport os\n\nfrom collections import defaultdict\n\n\nx = 1\n"

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        line_separator="\n",
    )
    config_remove = Config(remove_imports=["os", "collections"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert result_remove == "\nimport sys\n\nx = 1\n"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
    )
    config_no_imports = Config()
    result_no_imports = sorted_imports(parsed_no_imports, config_no_imports)
    assert result_no_imports == "x = 1\n"

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["", "x = 1", "y = 2"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from":


# LLM-generated content at query #15
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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
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
    config = Config(force_sort_within_sections=False)
    expected = "import os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 4: With custom config
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
    config = Config(
        force_sort_within_sections=True,
        lines_between_sections=1,
        lines_between_types=1,
    )
    expected = "import os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 5: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}},
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
    config = Config(
        import_headings={"future": "Future imports", "thirdparty": "Third party imports"},
        dedup_headings=True,
    )
    expected = "# Future imports\nfrom __future__ import annotations\n\n# Third party imports\nimport os\nimport sys\n\nfrom collections import OrderedDict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 6: With remove_imports
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
    config = Config(remove_imports=["os"])
    expected = "import sys\n\nfrom collections import OrderedDict\n"
    assert sorted_imports(parsed, config=config) == expected

    # Test case 7: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os", "import sys"]},
        import_placements={"# Placeholder": "THIRDPARTY"},
    )
    config = Config()
    expected = "# Placeholder\nimport os\nimport sys\n\nprint('hello')"
    assert sorted_imports(parsed, config=config) == expected


# LLM-generated content at query #16
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
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
    assert result == "from collections import defaultdict\nimport os\nimport sys\n"

    # Test case 2: No sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\nimport os\nimport sys\n"

    # Test case 3: Empty imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == ""

    # Test case 4: With import headings
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
    config = Config(import_headings={"thirdparty": "Third-party imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third-party imports\nfrom collections import defaultdict\nimport os\nimport sys\n"

    # Test case 5: With lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\n\nimport os\nimport sys\n"


# LLM-generated content at query #17
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            },
            "FIRSTPARTY": {
                "straight": {"my_module": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import my_module\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nprint('hello')"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": {"defaultdict"},
                    "os": {"*"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from os import *\nfrom collections import defaultdict\n\nprint('hello')"

    # Test with import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nprint('hello')"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(remove_imports=["os", "collections"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nprint('hello')"

    # Test with formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(formatting_function=lambda code, ext, cfg: code.replace("import os", "import os  # formatted"))
    result = sorted_imports(parsed, config)
    assert result == "import os  # formatted\n\nprint('hello')"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\n\nprint('hello')"

    # Test with lines_before_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\nprint('hello')"


# LLM-generated content at query #18
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result.startswith("import os\nimport zlib\n\nfrom collections import defaultdict\nfrom sys import path\n\nprint('hello')")

    # Test with no_sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": set(), "os": set()}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert result_no_sections.startswith("from __future__ import\nimport os\nimport zlib\n\nprint('hello')")

    # Test with forced_separate
    parsed_forced = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_forced = Config(forced_separate=["collections"])
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "from collections import defaultdict" in result_forced
    assert "from sys import path" in result_forced

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_remove = Config(remove_imports=["os", "sys"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" not in result_remove
    assert "from sys import path" not in result_remove
    assert "import zlib" in result_remove
    assert "from collections import defaultdict" in result_remove

    # Test with star_first
    parsed_star = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "sys": {"*": set(), "path": set()},
                    "collections": {"defaultdict": set()},
                },
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star.index("from sys import *") < result_star.index("from sys import path")

    # Test with from_first
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.startswith("from collections import defaultdict\nfrom sys import path\n\nimport os\nimport zlib\n\nprint('hello')")

    # Test with import_headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings

    # Test with lines_between_sections
    parsed_lines = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": set()}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": set(), "os": set()}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines.count("\n\n") >= 2

    # Test with lines_after_imports
    parsed_after = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_after = Config(lines_after_imports=2)
    result_after = sorted_imports(parsed_after, config_after)
    assert result_after.count("\n\n") >= 2

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    parsed_format = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": set(), "os": set()},
                "from": {"sys": {"path": set()}, "collections": {"defaultdict": set()}},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config_format = Config(formatting_function=custom_format)
    result_format = sorted_imports(parsed_format, config_format)
    assert "IMPORT OS" in result_format


# LLM-generated content at query #19
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result

    # Test with no sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "from __future__ import print_function" in result_no_sections
    assert "import os" in result_no_sections
    assert "import my_module" in result_no_sections

    # Test with forced separate
    parsed_forced = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"django": set()}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_app": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_forced = Config(forced_separate=["DJANGO"])
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "import django" in result_forced
    assert "import my_app" in result_forced

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": set(), "sys": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_remove = Config(remove_imports=["sys"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" in result_remove
    assert "import sys" not in result_remove

    # Test with from_first
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from sys import exit") < result_from_first.index("import os")

    # Test with star_first
    parsed_star = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "numpy": {"*"},
                    "os": {"path"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star.index("from numpy import *") < result_star.index("from os import path")

    # Test with import headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"requests": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings
    assert "import requests" in result_headings

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    result_no_imports = sorted_imports(parsed_no_imports)
    assert result_no_imports == "print('hello')\n"

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["# Place imports here", ""],
        imports={
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": []},
        import_placements={"# Place imports here": "THIRDPARTY"}
    )
    config_place = Config()
    result_place = sorted_imports(parsed_place, config_place)
    assert "# Place imports here" in result_place
    assert "import os" in result_place


# LLM-generated content at query #20
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(only_sections=False, reverse_sort=False)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]}
            },
            "FIRSTPARTY": {
                "straight": {"myapp": []},
                "from": {"myapp.utils": ["helper"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_no_sections = Config(no_sections=True, only_sections=False)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "import os\nimport zlib\nimport myapp" in result_no_sections
    assert "from json import load\nfrom sys import path\nfrom myapp.utils import helper" in result_no_sections

    # Test with remove_imports
    config_remove = Config(remove_imports=["os", "sys"])
    result_remove = sorted_imports(parsed, config_remove)
    assert "import os" not in result_remove
    assert "from sys import path" not in result_remove

    # Test with star_first=True
    parsed_star = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"numpy": ["*"], "pandas": ["DataFrame"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star == "from numpy import *\nfrom pandas import DataFrame\n"

    # Test with from_first=True
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed, config_from_first)
    assert result_from_first.startswith("from json import load\nfrom sys import path\n\nimport os\nimport zlib\n")

    # Test with lines_between_sections
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed, config_lines)
    assert result_lines.count("\n\n") >= 1

    # Test with import_headings
    config_headings = Config(import_headings={"thirdparty": "Third Party"})
    result_headings = sorted_imports(parsed, config_headings)
    assert "# Third Party" in result_headings

    # Test with dedup_headings=False
    config_dedup = Config(dedup_headings=False, import_headings={"thirdparty": "Third Party"})
    result_dedup = sorted_imports(parsed, config_dedup)
    assert result_dedup.count("# Third Party") == 1

    # Test with ensure_newline_before_comments=True
    parsed_comments = parse.ParsedContent(
        lines_without_imports=["# Comment"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_comments = Config(ensure_newline_before_comments=True)
    result_comments = sorted_imports(parsed_comments, config_comments)
    assert result_comments == "import os\n\n# Comment\n"

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    config_format = Config(formatting_function=custom_format)
    result_format = sorted_imports(parsed, config_format)
    assert result_format == "IMPORT OS\nIMPORT ZLIB\n\nFROM JSON IMPORT LOAD\nFROM SYS IMPORT PATH\n"

    # Test with lines_before_imports and lines_after_imports
    parsed_lines = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_lines_ba = Config(lines_before_imports=2, lines_after_imports=2)
    result_lines_ba = sorted_imports(parsed_lines, config_lines_ba)
    assert result_lines_ba.startswith("\n\nimport os\n\n\nprint('hello')")


# LLM-generated content at query #21
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\nx = 1"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    result = sorted_imports(parsed, config)
    assert result == "x = 1"

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {}
            },
            "STDLIB": {
                "straight": {"os": set()},
                "from": {"sys": {"exit"}}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    config = Config(import_headings={"future": "Future", "stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Future\nfrom __future__ import print_function\n\n# Standard Library\nimport os\n\nfrom sys import exit\n\n\nx = 1"

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"django": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    config = Config(forced_separate=["DJANGO"])
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport django\n\n\nx = 1"

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n"
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport sys\n\n\nx = 1"


# LLM-generated content at query #22
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 2: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "x = 1\n"

    # Test case 3: Remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 4: Force separate
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 5: No sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 6: Star first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["*"], "sys": ["path"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom collections import *\nfrom sys import path\n\nx = 1\n"

    # Test case 7: From first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test case 8: Import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test case 9: Import footers
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\n# End of Third Party Imports\n\nx = 1\n"

    # Test case 10: Lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=1,
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import annotations\n\n\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"


# LLM-generated content at query #23
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
    assert "# third party" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test no sections
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "# future" in result_no_sections
    assert "from __future__ import annotations" in result_no_sections
    assert "import os" in result_no_sections
    assert "import sys" in result_no_sections

    # Test remove imports
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
    config_remove = Config(remove_imports=["os"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" not in result_remove
    assert "import sys" in result_remove

    # Test empty imports
    parsed_empty = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result_empty = sorted_imports(parsed_empty)
    assert result_empty == "print('hello')\n"

    # Test star_first
    parsed_star = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_star = Config(star_first=True)
    result_star = sorted_imports(parsed_star, config_star)
    assert result_star.index("from module1 import *") < result_star.index("from module2 import func")

    # Test from_first
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
    assert result_from_first.index("from collections import defaultdict") < result_from_first.index("import os")

    # Test lines_between_sections
    parsed_lines = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines.count("\n\n") >= 2

    # Test reverse_sort
    parsed_reverse = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_reverse = Config(reverse_sort=True)
    result_reverse = sorted_imports(parsed_reverse, config_reverse)
    assert result_reverse.index("import sys") < result_reverse.index("import os")


# LLM-generated content at query #24
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
    assert sorted_imports(parsed) == ""

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
    assert sorted_imports(parsed) == "print('hello')"

    # Test case 3: Basic import sorting
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
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 4: With custom config
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
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 5: With sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "from __future__ import annotations" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 6: With no_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert "from __future__ import annotations" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test case 7: With remove_imports config
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
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os", "collections"])
    result = sorted_imports(parsed, config)
    assert "import os" not in result
    assert "import sys" in result
    assert "from collections import defaultdict" not in result
    assert "from itertools import chain" in result

    # Test case 8: With lines_between_types config
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
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_types=2)
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result
    assert result.count("") >= 2

    # Test case 9: With lines_between_sections config
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert "from __future__ import annotations" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result
    assert result.count("") >= 2

    # Test case 10: With import_headings config
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
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result
    assert "import os" in result
    assert "import sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result


# LLM-generated content at query #25
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed_content, config)
    assert result.startswith("import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef main():")
    assert result.endswith("pass")

    # Test case 2: No imports case
    parsed_content = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=2,
    )
    result = sorted_imports(parsed_content)
    assert result == "def main():\n    pass"

    # Test case 3: With sections and comments
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {"sys": ["argv"]}},
            "THIRDPARTY": {"straight": {}, "from": {"django": ["conf"]}}
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(
        import_headings={"future": "Future imports", "stdlib": "Standard library"},
        lines_between_sections=1,
    )
    result = sorted_imports(parsed_content, config)
    assert "# Future imports" in result
    assert "# Standard library" in result
    assert "from __future__ import annotations" in result
    assert "import os" in result
    assert "from sys import argv" in result
    assert "from django import conf" in result

    # Test case 4: With forced separate sections
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(forced_separate=["LOCALFOLDER"], lines_between_sections=1)
    parsed_content.imports["LOCALFOLDER"] = {"straight": {"local": []}, "from": {}}
    result = sorted_imports(parsed_content, config)
    assert "import numpy" in result
    assert "import my_module" in result
    assert "import local" in result

    # Test case 5: With remove_imports
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": [], "pandas": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(remove_imports=["numpy"])
    result = sorted_imports(parsed_content, config)
    assert "import numpy" not in result
    assert "import pandas" in result

    # Test case 6: With no_sections
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed_content, config)
    assert "import __future__" not in result
    assert "import os" in result
    assert "import numpy" in result

    # Test case 7: With star_first
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "numpy": ["*"],
                    "pandas": ["DataFrame"],
                    "scipy": ["*"],
                }
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed_content, config)
    lines = result.split("\n")
    star_lines = [i for i, line in enumerate(lines) if "*" in line]
    non_star_lines = [i for i, line in enumerate(lines) if "DataFrame" in line]
    assert star_lines < non_star_lines

    # Test case 8: With formatting_function
    def custom_format(code, extension, config):
        return code.replace("import", "from somewhere import")

    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed_content, config)
    assert "from somewhere import numpy" in result

    # Test case 9: With lines_before_imports and lines_after_imports
    parsed_content = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config(lines_before_imports=2, lines_after_imports=2)
    result = sorted_imports(parsed_content, config)
    lines = result.split("\n")
    assert lines[0] == ""
    assert lines[1] == ""
    assert lines[2] == "import numpy"
    assert lines[3] == ""
    assert lines[4] == ""
    assert lines[5] == "def main():"


# LLM-generated content at query #26
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
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
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 5: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=0,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import sys\n"
    )
    assert result == expected


# LLM-generated content at query #27
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
    config = Config()
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
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config()
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(forced_separate=["FUTURE"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            },
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FUTURE"],
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import annotations\n"
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 6: With reverse_sort
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import sys\n"
        "import os\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 7: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["defaultdict"],
                    "os": ["*"],
                    "sys": ["path"],
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
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from os import *\n"
        "from collections import defaultdict\n"
        "from sys import path\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 8: With from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 9: With import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
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
    expected = (
        "# Third Party Imports\n"
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 10: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": ["my_module"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY", "FIRSTPARTY"],
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "import my_module\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #28
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "x = 1"

    # Test with forced_separate
    config.forced_separate = ["LOCALFOLDER"]
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
            "LOCALFOLDER": {
                "straight": {"local": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert "import local" in result
    assert "import os" in result

    # Test with no_sections
    config.no_sections = True
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert "import __future__" in result
    assert "import os" in result

    # Test with only_sections
    config.only_sections = True
    config.no_sections = False
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    result = sorted_imports(parsed, config)
    assert "import os" in result
    assert "import sys" in result

    # Test with star_first
    config.star_first = True
    config.only_sections = False
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["*"],
                    "module2": ["func"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result.index("from module1 import *") < result.index("from module2 import func")

    # Test with from_first
    config.from_first = True
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    result = sorted_imports(parsed, config)
    assert result.index("from collections import defaultdict") < result.index("import os")

    # Test with import_headings
    config.import_headings = {"thirdparty": "Third Party Imports"}
    parsed = parse.ParsedContent(
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
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result

    # Test with lines_between_sections
    config.lines_between_sections = 2
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result.count("\n\n") >= 1

    # Test with lines_after_imports
    config.lines_after_imports = 2
    config.lines_between_sections = 1
    parsed = parse.ParsedContent(
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
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    import_end = lines.index("x = 1") - 1
    assert lines[import_end] == ""
    assert lines[import_end - 1] == ""

    # Test with remove_imports
    config.remove_imports = ["os"]
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert "import os" not in result
    assert "import sys" in result


# LLM-generated content at query #29
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict", "defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import OrderedDict, defaultdict\n\nx = 1\n"

    # Test case 2: No imports
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

    # Test case 3: With sections and headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(import_headings={"future": "Future", "stdlib": "Standard Library"})
    result = sorted_imports(parsed, config)
    assert "# Future\nfrom __future__ import __future__\n\n# Standard Library\nimport os\n\nx = 1\n" == result

    # Test case 4: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"django": ["django"], "flask": ["flask"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(forced_separate=["django"])
    result = sorted_imports(parsed, config)
    assert result == "import django\nimport flask\n\nx = 1\n"

    # Test case 5: With no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import __future__\nimport os\n\nx = 1\n"

    # Test case 6: With remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict", "defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nfrom collections import OrderedDict, defaultdict\n\nx = 1\n"

    # Test case 7: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module": ["*", "func"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module import *\nfrom module import func\n\nx = 1\n"

    # Test case 8: With from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import OrderedDict\n\nimport os\n\nx = 1\n"

    # Test case 9: With lines_between_types
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_between_types=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nfrom collections import OrderedDict\n\nx = 1\n"

    # Test case 10: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["__future__"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["os"]}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_between_sections=3)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import __future__\n\n\n\nimport os\n\nx = 1\n"


# LLM-generated content at query #30
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: No imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
        sections=[],
        imports={},
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test case 2: Simple imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')"
    )

    # Test case 3: No sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')"
    )

    # Test case 4: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["OrderedDict"]},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "import sys\n"
        "from collections import OrderedDict\n"
        "\n"
        "print('hello')"
    )

    # Test case 5: Lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY", "FIRSTPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            },
        },
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "\n"
        "\n"
        "import sys\n"
        "\n"
        "print('hello')"
    )

    # Test case 6: Import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == (
        "# Third Party Imports\n"
        "import os\n"
        "\n"
        "print('hello')"
    )

    # Test case 7: Lines after imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "\n"
        "\n"
        "print('hello')"
    )

    # Test case 8: Place imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')", "print('world')"],
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        sections=["THIRDPARTY"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        place_imports={"THIRDPARTY": ["import sys"]},
        import_placements={"print('hello')": "THIRDPARTY"},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == (
        "import os\n"
        "\n"
        "print('hello')\n"
        "import sys\n"
        "\n"
        "print('world')"
    )


# LLM-generated content at query #31
#--------------------------

```python
def test_sorted_imports():
    # Test basic import sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict", "OrderedDict"}}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport sys\n\nfrom collections import OrderedDict, defaultdict\n"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n"
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "print('hello')\n"

    # Test with forced separate sections
    config_forced = Config(forced_separate=["LOCALFOLDER"])
    parsed_forced = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}},
            "LOCALFOLDER": {"straight": {"sys": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_forced = sorted_imports(parsed_forced, config_forced)
    assert "import os\n" in result_forced
    assert "import sys\n" in result_forced

    # Test with star imports first
    config_star = Config(star_first=True)
    parsed_star = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": {"*"},
                    "module2": {"function1"}
                }
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_star = sorted_imports(parsed_star, config_star)
    assert "from module1 import *" in result_star
    assert "from module2 import function1" in result_star
    assert result_star.index("from module1 import *") < result_star.index("from module2 import function1")

    # Test with custom import headings
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed_headings = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings
    assert "import os" in result_headings

    # Test with lines between sections
    config_lines = Config(lines_between_sections=2)
    parsed_lines = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {"straight": {"os": set()}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert result_lines.count("\n\n") >= 2

    # Test with remove_imports
    config_remove = Config(remove_imports=["import sys"])
    parsed_remove = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" in result_remove
    assert "import sys" not in result_remove

    # Test with formatting function
    def custom_format(code, extension, config):
        return code.replace("import", "from __future__ import")

    config_format = Config(formatting_function=custom_format)
    parsed_format = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n"
    )
    result_format = sorted_imports(parsed_format, config_format)
    assert "from __future__ import os" in result_format


# LLM-generated content at query #32
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict, OrderedDict\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With config options
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import sys\n"
        "import os\n"
        "\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"], "sys": ["sys"]},
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": ["my_module"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import sys\n"
        "\n"
        "from collections import defaultdict, OrderedDict\n"
        "\n"
        "import my_module\n"
        "\n"
    )
    assert result == expected


# LLM-generated content at query #33
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
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import defaultdict" in result
    assert "from itertools import chain" in result

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "from __future__ import print_function" in result_no_sections
    assert "import os\nimport sys" in result_no_sections
    assert "from collections import defaultdict" in result_no_sections

    # Test with from_first=True
    parsed_from_first = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed_from_first, config_from_first)
    assert result_from_first.index("from sys import argv") < result_from_first.index("import os")

    # Test with star_first=True
    parsed_star_first = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"os": ["*"], "sys": ["argv"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_star_first = Config(star_first=True)
    result_star_first = sorted_imports(parsed_star_first, config_star_first)
    assert result_star_first.index("from os import *") < result_star_first.index("from sys import argv")

    # Test with force_sort_within_sections=True
    parsed_force_sort = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["argv"], "collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_force_sort = Config(force_sort_within_sections=True)
    result_force_sort = sorted_imports(parsed_force_sort, config_force_sort)
    assert result_force_sort.index("import os") < result_force_sort.index("import zlib")
    assert result_force_sort.index("from collections import defaultdict") < result_force_sort.index("from sys import argv")

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config_remove = Config(remove_imports=["sys", "from collections import defaultdict"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" in result_remove
    assert "import sys" not in result_remove
    assert "from collections import defaultdict" not in result_remove

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
        place_imports={},
        import_placements={},
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings
    assert result_headings.index("# Third Party Imports") < result_headings.index("import os")

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config_no_imports = Config()
    result_no_imports = sorted_imports(parsed_no_imports, config_no_imports)
    assert result_no_imports == "print('hello')"

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["# PLACEHOLDER", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"# PLACEHOLDER": "THIRDPARTY"},
    )
    config_place = Config()
    result_place = sorted_imports(parsed_place, config_place)
    assert result_place.index("# PLACEHOLDER") < result_place.index("import os")
    assert result_place.index("import os") < result_place.index("print('hello')")

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    parsed_format = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config_format = Config(formatting_function=custom_format)
    result_format = sorted_imports(parsed_format, config_format)
    assert "IMPORT OS" in result_format


# LLM-generated content at query #34
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    config = Config(force_sort_within_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    config = Config(forced_separate=["zlib", "sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nfrom json import load\n\nimport zlib\n\nfrom sys import path\n\nx = 1\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import *\nfrom json import load\n\nx = 1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import path\n\nimport os\n\nx = 1\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["os", "json"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n\nx = 1\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nfrom sys import path\n\nx = 1\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\n\nimport os\n\nx = 1\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nx = 1\n"

    # Test with formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.replace("\n", "\r\n")

    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=custom_format)
    result = sorted_imports(parsed, config)
    assert result == "import os\r\n\r\nx = 1\r\n"


# LLM-generated content at query #35
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
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
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": []}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "import __future__\nimport my_module\nimport os\nimport sys\n\nx = 1" in result_no_sections

    # Test with from_first=True
    config_from_first = Config(from_first=True)
    result_from_first = sorted_imports(parsed, config_from_first)
    assert result_from_first.startswith("from collections import defaultdict\n\nimport os\nimport sys")

    # Test with star_first=True
    parsed_star = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": ["*"],
                    "module2": ["function1"],
                    "module3": ["*"]
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
    result_star_first = sorted_imports(parsed_star, config_star_first)
    assert "from module1 import *" in result_star_first
    assert "from module3 import *" in result_star_first
    assert result_star_first.index("from module1 import *") < result_star_first.index("from module2 import function1")

    # Test with import_headings
    config_headings = Config(
        import_headings={"THIRDPARTY": "Third Party Imports"},
        dedup_headings=True
    )
    result_headings = sorted_imports(parsed, config_headings)
    assert "# Third Party Imports" in result_headings

    # Test with lines_between_sections
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed, config_lines)
    assert "\n\n\n" in result_lines

    # Test with ensure_newline_before_comments
    parsed_comments = parse.ParsedContent(
        lines_without_imports=["# comment", "x = 1"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config_comments = Config(ensure_newline_before_comments=True)
    result_comments = sorted_imports(parsed_comments, config_comments)
    assert "\n# comment" in result_comments

    # Test with formatting_function
    def custom_format(code, extension, config):
        return code.replace("import", "from custom import")
    config_formatting = Config(formatting_function=custom_format)
    result_formatting = sorted_imports(parsed, config_formatting)
    assert "from custom import os" in result_formatting

    # Test with place_imports
    parsed_place = parse.ParsedContent(
        lines_without_imports=["x = 1", "# PLACE_HOLDER"],
        imports={"THIRDPARTY": {"straight": {"os": []}, "from": {}}},
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"PLACE_HOLDER": ["import sys"]},
        import_placements={"# PLACE_HOLDER": "PLACE_HOLDER"}
    )
    result_place = sorted_imports(parsed_place, config)
    assert "import sys" in result_place
    assert result_place.index("import sys") > result_place.index("# PLACE_HOLDER")


# LLM-generated content at query #36
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]}
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
    assert result == "\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result == "print('hello')"

    # Test with forced separate
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]}
            },
            "LOCALFOLDER": {
                "straight": {"local": []},
                "from": {"": ["local_func"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert "import local" in result
    assert "from . import local_func" in result
    assert "import zlib" in result
    assert "from sys import path" in result

    # Test with no sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert "from __future__ import annotations" in result
    assert "import zlib" in result
    assert "from sys import path" in result

    # Test with remove imports
    config = Config(remove_imports=["from sys import *"])
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["*"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert "import zlib" in result
    assert "from sys import *" not in result

    # Test with star_first
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"sys": ["*"], "os": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result.index("from sys import *") < result.index("from os import path")

    # Test with from_first
    config = Config(from_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert result.index("from sys import path") < result.index("import zlib")

    # Test with import headings
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    assert "# Third Party Imports" in result
    assert "import zlib" in result

    # Test with lines_between_sections
    config = Config(lines_between_sections=2)
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["annotations"]},
                "from": {}
            },
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    future_line = lines.index("from __future__ import annotations")
    zlib_line = lines.index("import zlib")
    assert zlib_line - future_line == 3  # 2 empty lines + 1 line with import

    # Test with lines_after_imports
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    zlib_line = lines.index("import zlib")
    hello_line = lines.index("print('hello')")
    assert hello_line - zlib_line == 3  # 2 empty lines + 1 line with print

    # Test with place_imports
    config = Config()
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def func():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import zlib"]},
        import_placements={"def func():"}
    )
    result = sorted_imports(parsed, config)
    lines = result.split("\n")
    assert lines[0] == ""
    assert lines[1] == "def func():"
    assert lines[2] == "import zlib"
    assert lines[3] == "    pass"


# LLM-generated content at query #37
#--------------------------

```python
def test_sorted_imports():
    # Setup test data
    parsed_content = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["import annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": ["import os"]}, "from": {}},
            "THIRDPARTY": {"straight": {"django": ["import django"]}, "from": {}},
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )

    config = Config(
        remove_imports=[],
        forced_separate=[],
        no_sections=False,
        only_sections=False,
        reverse_sort=False,
        star_first=False,
        from_first=False,
        force_sort_within_sections=False,
        lines_between_types=0,
        lines_between_sections=1,
        import_headings={},
        import_footers={},
        dedup_headings=False,
        ensure_newline_before_comments=False,
        lines_before_imports=2,
        lines_after_imports=2,
        profile="black",
        section_comments=False,
        formatting_function=None,
    )

    # Call the function
    result = sorted_imports(parsed_content, config, "py", "import")

    # Expected output
    expected = (
        "\n"
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import os\n"
        "\n"
        "import django\n"
        "\n"
        "\n"
        "def main():\n"
        "    pass"
    )

    # Assertions
    assert result == expected


# LLM-generated content at query #38
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}, "itertools": {"chain"}},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed)
    assert result == (
        "import os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef foo():\n    pass\n"
    )

    # Test case 2: With forced_separate config
    config = Config(forced_separate=["LOCALFOLDER"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}},
            },
            "LOCALFOLDER": {
                "straight": {"my_module": set()},
                "from": {"my_package": {"my_function"}},
            },
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "import os\nimport sys\n\nfrom collections import defaultdict\n\n"
        "import my_module\n\nfrom my_package import my_function\n\n\ndef foo():\n    pass\n"
    )

    # Test case 3: With no_sections=True
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "FUTURE": {
                "straight": {"__future__": {"print_function"}},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"collections": {"defaultdict"}},
            },
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "from __future__ import print_function\n\nimport os\nimport sys\n\nfrom collections import defaultdict\n\n\ndef foo():\n    pass\n"
    )

    # Test case 4: With star_first=True
    config = Config(star_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "module1": {"*"},
                    "module2": {"function1", "function2"},
                },
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "from module1 import *\nfrom module2 import function1, function2\n\n\ndef foo():\n    pass\n"
    )

    # Test case 5: With from_first=True
    config = Config(from_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "from collections import defaultdict\n\nimport os\n\n\ndef foo():\n    pass\n"
    )

    # Test case 6: With lines_between_types=2
    config = Config(lines_between_types=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {"collections": {"defaultdict"}},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "import os\n\n\nfrom collections import defaultdict\n\n\ndef foo():\n    pass\n"
    )

    # Test case 7: With import_headings
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "# Third Party Imports\nimport os\n\n\ndef foo():\n    pass\n"
    )

    # Test case 8: With remove_imports
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "import sys\n\n\ndef foo():\n    pass\n"
    )

    # Test case 9: With lines_after_imports=2
    config = Config(lines_after_imports=2)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "import os\n\n\n\ndef foo():\n    pass\n"
    )

    # Test case 10: With formatting_function
    def custom_format(code: str, extension: str, config: Config) -> str:
        return code.upper()

    config = Config(formatting_function=custom_format)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config=config)
    assert result == (
        "IMPORT OS\n\n\ndef foo():\n    pass\n"
    )


# LLM-generated content at query #39
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom __future__ import annotations\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with forced separate
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(remove_imports=["import zlib", "from sys import path"])
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\n\nfrom json import load\n\nx = 1\n"

    # Test with star first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"sys": ["*"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom sys import *\nfrom json import load\n\nx = 1\n"

    # Test with from first
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom json import load\nfrom sys import path\n\nimport os\nimport zlib\n\nx = 1\n"

    # Test with import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Third Party Imports\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nfrom __future__ import annotations\n\n\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with lines before imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\n\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\nx = 1\n"

    # Test with lines after imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n\n\n\nx = 1\n"

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


# LLM-generated content at query #40
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=[], forced_separate=[], no_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport os\nimport sys\n"

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": []}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=[], forced_separate=[], no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\nimport my_module\nimport os\nimport sys\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=["os"], forced_separate=[], no_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nimport sys\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func"]}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(remove_imports=[], forced_separate=[], no_sections=False, star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import func\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(
        remove_imports=[],
        forced_separate=[],
        no_sections=False,
        import_headings={"thirdparty": "Third Party Imports"}
    )
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}}
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={}
    )
    config = Config(
        remove_imports=[],
        forced_separate=[],
        no_sections=False,
        lines_between_sections=2
    )
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import absolute_import\n\n\nimport os\n"

    # Test with place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
        place_imports={"THIRDPARTY": ["import os"]},
        import_placements={"def main():"}
    )
    config = Config(remove_imports=[], forced_separate=[], no_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "def main():\nimport os\n    pass\n"


# LLM-generated content at query #41
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

    # Test with forced_separate
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
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

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
    config = Config(remove_imports=["os", "sys"])
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\n"

    # Test with lines_between_types
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
    config = Config(lines_between_types=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\n\nimport os\nimport sys\n"

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
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

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

    # Test with import_footers
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
    config = Config(import_footers={"thirdparty": "End of Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n# End of Third Party Imports\n"

    # Test with lines_between_sections
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
    config = Config(lines_between_sections=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with ensure_newline_before_comments
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
    config = Config(ensure_newline_before_comments=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with formatting_function
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
    config = Config(formatting_function=lambda x, y, z: x)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with lines_before_imports
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
    config = Config(lines_before_imports=1)
    result = sorted_imports(parsed, config)
    assert result == "\nfrom collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with lines_after_imports
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
    config = Config(lines_after_imports=1)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n\n"




# LLM-generated content at query #42
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with default config
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result.startswith("import os\nimport sys\nfrom collections import defaultdict\nfrom itertools import chain\n\n\ndef main():")

    # Test case 2: No imports to sort
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=2,
    )
    result = sorted_imports(parsed, config)
    assert result == "def main():\n    pass"

    # Test case 3: With custom config (reverse sort)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(reverse_sort=True)
    result = sorted_imports(parsed, config)
    assert result.startswith("import sys\nimport os\nfrom itertools import chain\nfrom collections import defaultdict\n\n\ndef main():")

    # Test case 4: With star imports and star_first=True
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func1"]}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result.startswith("from module1 import *\nfrom module2 import func1\n\n\ndef main():")

    # Test case 5: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def main():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {}
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result.startswith("# Third Party Imports\nimport os\n\n\ndef main():")


# LLM-generated content at query #43
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting of imports
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
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with no_sections=True
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            },
            "FIRSTPARTY": {
                "straight": {"my_module": []},
                "from": {"my_package": ["MyClass"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nfrom my_package import MyClass\nimport my_module\nimport os\nimport sys\n"

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
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\nfrom itertools import chain\nimport os\nimport sys\n"

    # Test with star_first=True
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import function\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n"

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {
                "straight": {"__future__": ["print_function"]},
                "from": {},
            },
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import print_function\n\n\nimport os\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(remove_imports=["os"])
    result = sorted_imports(parsed, config)
    assert result == "import sys\n"

    # Test with no imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"


# LLM-generated content at query #44
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={"FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}}},
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function"

    # Test case 2: Sorting with sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(no_sections=False)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function\n\nimport os\nimport sys"

    # Test case 3: Remove imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "THIRDPARTY": {"straight": {"django": [], "flask": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(remove_imports=["django"])
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function\n\nimport flask"

    # Test case 4: Force separate
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(forced_separate=["os"])
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function\n\nimport os\n\nimport sys"

    # Test case 5: Lines between sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function\n\n\n\nimport os\nimport sys"

    # Test case 6: From first
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {}, "from": {"__future__": ["print_function"]}},
            "STDLIB": {"straight": {"os": []}, "from": {"sys": ["argv"]}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom __future__ import print_function\nfrom sys import argv\n\nimport os"

    # Test case 7: Star first
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {}, "from": {}},
            "STDLIB": {"straight": {}, "from": {"os": ["*"], "sys": ["argv"]}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nfrom os import *\nfrom sys import argv"

    # Test case 8: Import headings
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(import_headings={"future": "Future imports", "stdlib": "Standard library"})
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\n# Future imports\nfrom __future__ import print_function\n\n# Standard library\nimport os"

    # Test case 9: Force sort within sections
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {}, "from": {}},
            "STDLIB": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(force_sort_within_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "# Test file\n\nimport os\nimport sys"

    # Test case 10: Lines before imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Test file", ""],
        imports={
            "FUTURE": {"straight": {"__future__": ["print_function"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {}},
        },
        import_index=1,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_before_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "\n\n# Test file\n\nfrom __future__ import print_function\n\nimport os"


# LLM-generated content at query #45
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
        sections=[],
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
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config()
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
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
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
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(forced_separate=["FUTURE"])
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
                "straight": {"os": ["os.path"], "sys": ["sys.path"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        place_imports={},
        import_placements={},
        sections=["THIRDPARTY"],
        original_line_count=1,
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = """import os
import sys

from collections import defaultdict

print('hello')"""
    assert result == expected


# LLM-generated content at query #46
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

    # Test with no_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import annotations\nimport os\nimport zlib\n"

    # Test with forced_separate
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
            "FIRSTPARTY": {"straight": {"my_module": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(forced_separate=["LOCALFOLDER"])
    result = sorted_imports(parsed, config)
    assert result == "import os\nimport zlib\n\nimport my_module\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"zlib": [], "os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["zlib"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {"module1": ["*"], "module2": ["func"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from module1 import *\nfrom module2 import func\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": []},
                "from": {"sys": ["path"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from sys import path\n\nimport os\n"

    # Test with import_headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
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
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "from __future__ import annotations\n\n\nimport os\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["def main():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\ndef main():\n    pass\n"

    # Test with formatting_function
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(formatting_function=lambda code, ext, cfg: code.replace("import", "from"))
    result = sorted_imports(parsed, config)
    assert result == "from os\n"


# LLM-generated content at query #47
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
                "from": {"collections": ["defaultdict", "OrderedDict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict, OrderedDict\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 4: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {"typing": ["List"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(forced_separate=["FIRSTPARTY"])
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "import os\n"
        "\n"
        "from typing import List\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 5: With no sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {"typing": ["List"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "from typing import List\n"
        "import os\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 6: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["*"],
                    "os": ["path"],
                    "sys": ["*"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import *\n"
        "from sys import *\n"
        "from os import path\n"
    )
    assert result == expected

    # Test case 7: With from_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {"collections": ["defaultdict"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from collections import defaultdict\n"
        "\n"
        "import os\n"
    )
    assert result == expected

    # Test case 8: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
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
    expected = (
        "# Third Party Imports\n"
        "from collections import defaultdict\n"
        "import os\n"
    )
    assert result == expected

    # Test case 9: With lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            },
            "FIRSTPARTY": {
                "straight": {"sys": ["sys"]},
                "from": {},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "\n"
        "import sys\n"
    )
    assert result == expected

    # Test case 10: With lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["os"]},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #48
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
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
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"
    assert result == expected

    # Test case 4: With custom config
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        reverse_sort=True,
        from_first=True,
        lines_between_sections=2,
    )
    result = sorted_imports(parsed, config)
    expected = "from sys import path\nfrom json import load\n\nimport zlib\nimport os\n"
    assert result == expected

    # Test case 5: With forced separate sections
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        forced_separate=["zlib", "os"],
    )
    result = sorted_imports(parsed, config)
    expected = "import os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"
    assert result == expected

    # Test case 6: With no sections
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        no_sections=True,
    )
    result = sorted_imports(parsed, config)
    expected = "import json\nimport os\nimport sys\nimport zlib\n"
    assert result == expected

    # Test case 7: With star_first
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "from": {"sys": ["*"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        star_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = "from sys import *\nfrom json import load\n"
    assert result == expected

    # Test case 8: With import headings
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    expected = "# Third Party Imports\nimport os\nimport zlib\n\nfrom json import load\nfrom sys import path\n"
    assert result == expected

    # Test case 9: With formatting function
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        formatting_function=lambda code, ext, cfg: code.upper(),
    )
    result = sorted_imports(parsed, config)
    expected = "IMPORT OS\nIMPORT ZLIB\n\nFROM JSON IMPORT LOAD\nFROM SYS IMPORT PATH\n"
    assert result == expected

    # Test case 10: With place_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["# Placeholder", "print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": [], "os": []},
                "from": {"sys": ["path"], "json": ["load"]},
            }
        },
        import_index=0,
        original_line_count=2,
        line_separator="\n",
        place_imports={"# Placeholder": ["import os", "import zlib"]},
        import_placements={"# Placeholder": "# Placeholder"},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = "# Placeholder\nimport os\nimport zlib\n\nprint('hello')\n"
    assert result == expected


# LLM-generated content at query #49
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    assert result == "import os\nimport sys\n\nfrom collections import defaultdict\n\nx = 1\n"

    # Test with no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": [], "sys": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\nimport os\nimport sys\n\nx = 1\n"

    # Test with from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
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
    config = Config(from_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import defaultdict\n\nimport os\n\nx = 1\n"

    # Test with star_first
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {},
                "from": {
                    "collections": ["*"],
                    "os": ["path"],
                },
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    assert result == "from collections import *\nfrom os import path\n\nx = 1\n"

    # Test with import headings
    parsed = parse.ParsedContent(
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
    config = Config(import_headings={"thirdparty": "Third Party Imports"})
    result = sorted_imports(parsed, config)
    assert result == "# Third Party Imports\nimport os\n\nx = 1\n"

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

    # Test with lines_between_sections
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": []}, "from": {}},
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_between_sections=2)
    result = sorted_imports(parsed, config)
    assert result == "import __future__\n\n\nimport os\n\nx = 1\n"

    # Test with lines_after_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {"straight": {"os": []}, "from": {}},
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(lines_after_imports=2)
    result = sorted_imports(parsed, config)
    assert result == "import os\n\n\nx = 1\n"

    # Test with remove_imports
    parsed = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config(remove_imports=["sys"])
    result = sorted_imports(parsed, config)
    assert result == "import os\n\nx = 1\n"


# LLM-generated content at query #50
#--------------------------

```python
def test_sorted_imports():
    # Test case 1: Basic sorting with no sections
    config = Config(no_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["import annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": ["import os"], "sys": ["import sys"]}, "from": {}},
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\n\ndef foo():\n    pass"

    # Test case 2: Sorting with sections and custom config
    config = Config(
        lines_between_sections=1,
        lines_between_types=1,
        import_headings={"FUTURE": "Future Imports"},
    )
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "FUTURE": {"straight": {"__future__": ["import annotations"]}, "from": {}},
            "THIRDPARTY": {"straight": {"os": ["import os"], "sys": ["import sys"]}, "from": {}},
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "\n# Future Imports\nfrom __future__ import annotations\n\nimport os\nimport sys\n\ndef foo():\n    pass"

    # Test case 3: Remove imports
    config = Config(remove_imports=["os"])
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {"straight": {"os": ["import os"], "sys": ["import sys"]}, "from": {}},
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport sys\n\ndef foo():\n    pass"

    # Test case 4: Force sort within sections
    config = Config(force_sort_within_sections=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"numpy": ["import numpy as np"], "pandas": ["import pandas as pd"]},
            },
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "\nimport os\nimport sys\nfrom numpy import numpy as np\nfrom pandas import pandas as pd\n\ndef foo():\n    pass"

    # Test case 5: From first
    config = Config(from_first=True)
    parsed = parse.ParsedContent(
        lines_without_imports=["", "def foo():", "    pass"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": ["import os"], "sys": ["import sys"]},
                "from": {"numpy": ["import numpy as np"], "pandas": ["import pandas as pd"]},
            },
        },
        import_index=0,
        original_line_count=3,
        line_separator="\n",
    )
    result = sorted_imports(parsed, config)
    assert result == "\nfrom numpy import numpy as np\nfrom pandas import pandas as pd\n\nimport os\nimport sys\n\ndef foo():\n    pass"


# LLM-generated content at query #51
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"itertools": ["chain"], "collections": ["abc"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert "import os\nimport sys" in result
    assert "from collections import abc\nfrom itertools import chain" in result

    # Test with no_sections=True
    parsed_no_sections = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"itertools": ["chain"], "collections": ["abc"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_sections = Config(no_sections=True)
    result_no_sections = sorted_imports(parsed_no_sections, config_no_sections)
    assert "import os\nimport sys" in result_no_sections
    assert "from collections import abc\nfrom itertools import chain" in result_no_sections

    # Test with remove_imports
    parsed_remove = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"itertools": ["chain"], "collections": ["abc"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_remove = Config(remove_imports=["os"])
    result_remove = sorted_imports(parsed_remove, config_remove)
    assert "import os" not in result_remove
    assert "import sys" in result_remove

    # Test with lines_between_sections
    parsed_lines = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"itertools": ["chain"], "collections": ["abc"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_lines = Config(lines_between_sections=2)
    result_lines = sorted_imports(parsed_lines, config_lines)
    assert "\n\n" in result_lines

    # Test with import_headings
    parsed_headings = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"itertools": ["chain"], "collections": ["abc"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
    )
    config_headings = Config(import_headings={"thirdparty": "Third Party Imports"})
    result_headings = sorted_imports(parsed_headings, config_headings)
    assert "# Third Party Imports" in result_headings

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={},
        import_index=-1,
        original_line_count=1,
        line_separator="\n",
    )
    config_no_imports = Config()
    result_no_imports = sorted_imports(parsed_no_imports, config_no_imports)
    assert result_no_imports == "print('hello')"


# LLM-generated content at query #52
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
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    assert result == "print('hello')\n"

    # Test case 3: Simple import sorting
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
        place_imports={},
        import_placements={},
    )
    config = DEFAULT_CONFIG
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected

    # Test case 4: With custom config
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
        place_imports={},
        import_placements={},
    )
    config = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "from json import load\n"
        "from sys import path\n"
        "\n"
        "import os\n"
        "import zlib\n"
        "\n"
        "\n"
    )
    assert result == expected

    # Test case 5: With forced separate sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {"json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        forced_separate=["FIRSTPARTY"],
        lines_between_sections=1,
    )
    result = sorted_imports(parsed, config)
    expected = (
        "import zlib\n"
        "\n"
        "from sys import path\n"
        "\n"
        "\n"
        "import os\n"
        "\n"
        "from json import load\n"
        "\n"
    )
    assert result == expected

    # Test case 6: With star imports first
    parsed = parse.ParsedContent(
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
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from sys import *\n"
        "from json import load\n"
        "\n"
    )
    assert result == expected

    # Test case 7: With import headings
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            }
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(
        import_headings={"thirdparty": "Third Party Imports"},
    )
    result = sorted_imports(parsed, config)
    expected = (
        "# Third Party Imports\n"
        "import zlib\n"
        "\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected

    # Test case 8: With no sections
    parsed = parse.ParsedContent(
        lines_without_imports=[""],
        imports={
            "THIRDPARTY": {
                "straight": {"zlib": []},
                "from": {"sys": ["path"]},
            },
            "FIRSTPARTY": {
                "straight": {"os": []},
                "from": {"json": ["load"]},
            },
        },
        import_index=0,
        original_line_count=1,
        line_separator="\n",
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True)
    result = sorted_imports(parsed, config)
    expected = (
        "import os\n"
        "import zlib\n"
        "\n"
        "from json import load\n"
        "from sys import path\n"
        "\n"
    )
    assert result == expected


# LLM-generated content at query #53
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
                "from": {"os.path": {"join": set()}},
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
        "from os.path import join\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 3: With sections and comments
    parsed = parse.ParsedContent(
        lines_without_imports=["# Main code", "print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"os.path": {"join": set()}},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=3,
        place_imports={},
        import_placements={},
    )
    config = Config(import_headings={"future": "Future imports"})
    result = sorted_imports(parsed, config)
    expected = (
        "# Future imports\n"
        "from __future__ import print_function\n"
        "\n"
        "import os\n"
        "import sys\n"
        "from os.path import join\n"
        "\n"
        "# Main code\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 4: With forced separate and no sections
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "FUTURE": {"straight": {"__future__": {"print_function"}}, "from": {}},
            "THIRDPARTY": {
                "straight": {"os": set(), "sys": set()},
                "from": {"os.path": {"join": set()}},
            },
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(no_sections=True, forced_separate=["FUTURE"])
    result = sorted_imports(parsed, config)
    expected = (
        "from __future__ import print_function\n"
        "\n"
        "import os\n"
        "import sys\n"
        "from os.path import join\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected

    # Test case 5: With star_first and from_first
    parsed = parse.ParsedContent(
        lines_without_imports=["print('hello')"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": set()},
                "from": {
                    "os.path": {"join": set()},
                    "module": {"*"},
                },
            }
        },
        import_index=0,
        line_separator="\n",
        original_line_count=2,
        place_imports={},
        import_placements={},
    )
    config = Config(star_first=True, from_first=True)
    result = sorted_imports(parsed, config)
    expected = (
        "from module import *\n"
        "from os.path import join\n"
        "\n"
        "import os\n"
        "\n"
        "print('hello')\n"
    )
    assert result == expected


# LLM-generated content at query #54
#--------------------------

```python
def test_sorted_imports():
    # Test basic sorting
    parsed = parse.ParsedContent(
        lines_without_imports=["# test", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
    )
    config = Config()
    result = sorted_imports(parsed, config)
    assert result == "# test\n\nimport os\nimport sys\n\nfrom collections import defaultdict\nfrom itertools import chain\n\nx = 1"

    # Test with no imports
    parsed_no_imports = parse.ParsedContent(
        lines_without_imports=["x = 1"],
        imports={},
        import_index=-1,
        line_separator="\n",
        original_line_count=1,
    )
    result_no_imports = sorted_imports(parsed_no_imports, config)
    assert result_no_imports == "x = 1"

    # Test with config options
    config_custom = Config(
        lines_between_sections=2,
        lines_between_types=1,
        from_first=True,
        reverse_sort=True,
    )
    parsed_custom = parse.ParsedContent(
        lines_without_imports=["# test", "x = 1"],
        imports={
            "THIRDPARTY": {
                "straight": {"os": [], "sys": []},
                "from": {"collections": ["defaultdict"], "itertools": ["chain"]},
            }
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
    )
    result_custom = sorted_imports(parsed_custom, config_custom)
    assert result_custom == "# test\n\nfrom itertools import chain\nfrom collections import defaultdict\n\nimport sys\nimport os\n\nx = 1"

    # Test with sections
    parsed_sections = parse.ParsedContent(
        lines_without_imports=["# test", "x = 1"],
        imports={
            "FUTURE": {"straight": {"__future__": ["annotations"]}, "from": {}},
            "STDLIB": {"straight": {"os": []}, "from": {"collections": ["defaultdict"]}},
            "THIRDPARTY": {"straight": {"numpy": []}, "from": {}},
        },
        import_index=1,
        line_separator="\n",
        original_line_count=2,
    )
    result_sections = sorted_imports(parsed_sections, config)
    assert result_sections == "# test\n\nfrom __future__ import annotations\n\nimport os\n\nfrom collections import defaultdict\n\nimport numpy\n\nx = 1"


