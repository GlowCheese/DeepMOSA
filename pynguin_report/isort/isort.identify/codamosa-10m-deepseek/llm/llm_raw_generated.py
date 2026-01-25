####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import_instance = Import(line_number=42, indented=False, module="os", file_path=Path("example.py"))
    assert str(import_instance) == "example.py:42 import os"

    import_instance = Import(line_number=10, indented=True, module="math", attribute="pi", alias="PI", file_path=Path("example.py"))
    assert str(import_instance) == "example.py:10 indented from math import pi as PI"

    import_instance = Import(line_number=7, indented=False, module="sys", cimport=True, file_path=Path("example.py"))
    assert str(import_instance) == "example.py:7 cimport sys"


# LLM-generated content at query #2
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import_instance = Import(line_number=1, indented=False, module="os", file_path=Path("test.py"))
    assert str(import_instance) == "test.py:1 import os"

    import_instance = Import(line_number=2, indented=True, module="sys", file_path=Path("test.py"))
    assert str(import_instance) == "test.py:2 indented import sys"

    import_instance = Import(line_number=3, indented=False, module="math", attribute="sqrt", file_path=Path("test.py"))
    assert str(import_instance) == "test.py:3 from math import sqrt"

    import_instance = Import(line_number=4, indented=False, module="numpy", alias="np", file_path=Path("test.py"))
    assert str(import_instance) == "test.py:4 import numpy as np"

    import_instance = Import(line_number=5, indented=True, module="pandas", attribute="DataFrame", alias="df", file_path=Path("test.py"))
    assert str(import_instance) == "test.py:5 indented from pandas import DataFrame as df"

    import_instance = Import(line_number=6, indented=False, module="cython", cimport=True, file_path=Path("test.py"))
    assert str(import_instance) == "test.py:6 cimport cython"

    import_instance = Import(line_number=7, indented=True, module="cython", attribute="cfunc", cimport=True, file_path=Path("test.py"))
    assert str(import_instance) == "test.py:7 indented from cython cimport cfunc"

    import_instance = Import(line_number=8, indented=False, module="requests", file_path=None)
    assert str(import_instance) == ":8 import requests"


# LLM-generated content at query #3
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO
    from pathlib import Path

    config = DEFAULT_CONFIG

    # Test with a simple import statement
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None

    # Test with a from import statement
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None

    # Test with an aliased import statement
    input_stream = StringIO("import os as operating_system\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias == "operating_system"

    # Test with a from import statement with an alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"

    # Test with a multiline import statement
    input_stream = StringIO("from os import (\n    path,\n    sep\n)\n")
    result = list(imports(input_stream, config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "sep"

    # Test with a cimport statement
    input_stream = StringIO("cimport numpy as np\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].attribute is None
    assert result[0].alias == "np"
    assert result[0].cimport is True

    # Test with a file path
    file_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, config, file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test with top_only flag
    input_stream = StringIO("import os\ndef foo():\n    pass\n")
    result = list(imports(input_stream, config, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test with a redundant alias
    config = DEFAULT_CONFIG._replace(remove_redundant_aliases=True)
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test with a redundant alias in from import
    input_stream = StringIO("from os import path as path\n")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None


# LLM-generated content at query #4
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import_obj = Import(line_number=1, indented=False, module='os', attribute=None, alias=None, cimport=False, file_path=None)
    assert str(import_obj) == ":1 import os"

    import_obj = Import(line_number=2, indented=True, module='sys', attribute='path', alias='p', cimport=False, file_path=Path('test.py'))
    assert str(import_obj) == "test.py:2 indented from sys import path as p"

    import_obj = Import(line_number=3, indented=False, module='numpy', attribute=None, alias='np', cimport=True, file_path=Path('test.py'))
    assert str(import_obj) == "test.py:3 cimport numpy as np"

    import_obj = Import(line_number=4, indented=True, module='pandas', attribute='DataFrame', alias=None, cimport=False, file_path=None)
    assert str(import_obj) == ":4 indented from pandas import DataFrame"


# LLM-generated content at query #5
#--------------------------

# Unit test for method statement of class Import
def test_Import_statement():
    assert Import(1, False, "module", None, None).statement() == "import module"
    assert Import(1, False, "module", "attribute", None).statement() == "from module import attribute"
    assert Import(1, False, "module", "attribute", "alias").statement() == "from module import attribute as alias"
    assert Import(1, False, "module", None, "alias").statement() == "import module as alias"
    assert Import(1, False, "module", None, None, True).statement() == "cimport module"
    assert Import(1, False, "module", "attribute", None, True).statement() == "from module cimport attribute"
    assert Import(1, False, "module", "attribute", "alias", True).statement() == "from module cimport attribute as alias"
    assert Import(1, False, "module", None, "alias", True).statement() == "cimport module as alias"


# LLM-generated content at query #6
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test basic import
    input_stream = io.StringIO("import os\nimport sys")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test from import
    input_stream = io.StringIO("from os import path")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test aliased import
    input_stream = io.StringIO("import pandas as pd")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "pandas"
    assert imports_list[0].alias == "pd"

    # Test cimport
    input_stream = io.StringIO("cimport numpy as np")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"
    assert imports_list[0].cimport is True

    # Test indented import
    input_stream = io.StringIO("    import os")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].indented is True

    # Test file path
    test_path = Path("test.py")
    input_stream = io.StringIO("import os")
    imports_list = list(imports(input_stream, file_path=test_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == test_path

    # Test top_only
    input_stream = io.StringIO("import os\ndef foo():\n    import sys")
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #7
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\nfrom sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].attribute == "path"
    assert result[1].module == "sys" and result[1].attribute == "version"

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system\nfrom sys import version as v")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].alias == "operating_system"
    assert result[1].module == "sys" and result[1].attribute == "version" and result[1].alias == "v"

    # Test case 4: Cimport
    input_stream = io.StringIO("cimport numpy as np\nfrom cython cimport boundscheck")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy" and result[0].alias == "np" and result[0].cimport
    assert result[1].module == "cython" and result[1].attribute == "boundscheck" and result[1].cimport

    # Test case 5: Indented import
    input_stream = io.StringIO("def foo():\n    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os" and result[0].indented

    # Test case 6: Multi-line import
    input_stream = io.StringIO("from os import (\n    path,\n    name\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].attribute == "path"
    assert result[1].module == "os" and result[1].attribute == "name"

    # Test case 7: Top only
    input_stream = io.StringIO("import os\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 8: File path
    file_path = Path("test.py")
    input_stream = io.StringIO("import os")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #8
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test simple straight import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test simple cimport
    input_stream = StringIO("cimport numpy\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is True
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test simple from import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is True

    # Test multiple imports in one line
    input_stream = StringIO("import os, sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "sys"
    assert result[1].attribute is None
    assert result[1].alias is None
    assert result[1].cimport is False
    assert result[1].line_number == 1
    assert result[1].indented is False

    # Test multiple from imports in one line
    input_stream = StringIO("from os import path, environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias is None
    assert result[1].cimport is False
    assert result[1].line_number == 1
    assert result[1].indented is False

    # Test import with alias
    input_stream = StringIO("import os as operating_system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias == "operating_system"
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test top_only parameter
    input_stream = StringIO("import os\n\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False

    # Test commented import
    input_stream = StringIO("# import os\n")
    result = list(imports(input_stream))
    assert len(result) == 0

    # Test multiline import
    input_stream = StringIO("from os import (path, environ)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias is None
    assert result[1].cimport is False
    assert result[1].line_number == 1
    assert result[1].indented is False

    # Test multiline import with alias
    input_stream = StringIO("from os import (path as p, environ as e)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias == "e"
    assert result[1].cimport is False
    assert result[1].line_number == 1
    assert result[1].indented is False

    # Test multiline import with continuation
    input_stream = StringIO("from os import path, \\\n    environ\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias is None
    assert result[1].cimport is False
    assert result[1].line_number == 2
    assert result[1].indented is False

    # Test multiline import with continuation and alias
    input_stream = StringIO("from os import path as p, \\\n    environ as e\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[0].cimport is False
    assert result[0].line_number == 1
    assert result[0].indented is False
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias == "e"
    assert result[1].cimport is False
    assert result[1].line_number == 2
    assert result[1].indented is False


# LLM-generated content at query #9
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import1 = Import(1, False, "module1", None, None, False, None)
    assert str(import1) == ":1 import module1"
    
    import2 = Import(2, True, "module2", "attr2", None, False, Path("test.py"))
    assert str(import2) == "test.py:2 indented from module2 import attr2"
    
    import3 = Import(3, False, "module3", None, "alias3", True, Path("test.py"))
    assert str(import3) == "test.py:3 cimport module3 as alias3"
    
    import4 = Import(4, True, "module4", "attr4", "alias4", False, None)
    assert str(import4) == ":4 indented from module4 import attr4 as alias4"


# LLM-generated content at query #10
#--------------------------

# Unit test for method statement of class Import
def test_Import_statement():
    # Test basic import statement
    import_stmt = Import(1, False, "module")
    assert import_stmt.statement() == "import module"

    # Test from import statement
    from_import_stmt = Import(1, False, "module", "attribute")
    assert from_import_stmt.statement() == "from module import attribute"

    # Test import with alias
    import_with_alias = Import(1, False, "module", alias="alias")
    assert import_with_alias.statement() == "import module as alias"

    # Test from import with alias
    from_import_with_alias = Import(1, False, "module", "attribute", "alias")
    assert from_import_with_alias.statement() == "from module import attribute as alias"

    # Test cimport statement
    cimport_stmt = Import(1, False, "module", cimport=True)
    assert cimport_stmt.statement() == "cimport module"

    # Test from cimport statement
    from_cimport_stmt = Import(1, False, "module", "attribute", cimport=True)
    assert from_cimport_stmt.statement() == "from module cimport attribute"

    # Test cimport with alias
    cimport_with_alias = Import(1, False, "module", alias="alias", cimport=True)
    assert cimport_with_alias.statement() == "cimport module as alias"

    # Test from cimport with alias
    from_cimport_with_alias = Import(1, False, "module", "attribute", "alias", cimport=True)
    assert from_cimport_with_alias.statement() == "from module cimport attribute as alias"


# LLM-generated content at query #11
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\nfrom sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].attribute == "path"
    assert result[1].module == "sys" and result[1].attribute == "version"

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system\nfrom sys import version as ver")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].alias == "operating_system"
    assert result[1].module == "sys" and result[1].attribute == "version" and result[1].alias == "ver"

    # Test case 4: Cimport
    input_stream = io.StringIO("cimport numpy as np\nfrom numpy cimport array")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy" and result[0].alias == "np" and result[0].cimport
    assert result[1].module == "numpy" and result[1].attribute == "array" and result[1].cimport

    # Test case 5: Multi-line import
    input_stream = io.StringIO("from os import (\n    path,\n    name\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].attribute == "path"
    assert result[1].module == "os" and result[1].attribute == "name"

    # Test case 6: File path included
    file_path = Path("test.py")
    input_stream = io.StringIO("import test_module")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].module == "test_module" and result[0].file_path == file_path

    # Test case 7: Top only
    input_stream = io.StringIO("import os\ndef function():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #12
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 3: Aliased import
    input_stream = io.StringIO("import numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test case 4: Cimport
    input_stream = io.StringIO("cimport numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport is True

    # Test case 5: Multi-line import
    input_stream = io.StringIO("from os import (\n    path,\n    name\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "name"

    # Test case 6: Top only
    input_stream = io.StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 7: With file path
    input_stream = io.StringIO("import os\n")
    result = list(imports(input_stream, file_path=Path("test.py")))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].file_path == Path("test.py")

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #13
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    config = Config()
    file_path = Path("example.py")

    # Test case 1: Basic import
    input_stream = io.StringIO("import os\nimport sys")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", file_path=file_path),
        Import(line_number=2, indented=False, module="sys", file_path=file_path)
    ]

    # Test case 2: From import with attribute
    input_stream = io.StringIO("from os import path")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", attribute="path", file_path=file_path)
    ]

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", alias="operating_system", file_path=file_path)
    ]

    # Test case 4: From import with multiple attributes
    input_stream = io.StringIO("from os import path, sep")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", attribute="path", file_path=file_path),
        Import(line_number=1, indented=False, module="os", attribute="sep", file_path=file_path)
    ]

    # Test case 5: Indented import
    input_stream = io.StringIO("    import os")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=True, module="os", file_path=file_path)
    ]

    # Test case 6: Cimport
    input_stream = io.StringIO("cimport numpy as np")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="numpy", alias="np", cimport=True, file_path=file_path)
    ]

    # Test case 7: Top only with statement declaration
    input_stream = io.StringIO("import os\ndef foo():\n    import sys")
    assert list(imports(input_stream, config, file_path, top_only=True)) == [
        Import(line_number=1, indented=False, module="os", file_path=file_path)
    ]

    # Test case 8: Import with continuation line
    input_stream = io.StringIO("import os, \\\n    sys")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", file_path=file_path),
        Import(line_number=2, indented=False, module="sys", file_path=file_path)
    ]

    # Test case 9: From import with continuation line
    input_stream = io.StringIO("from os import path, \\\n    sep")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", attribute="path", file_path=file_path),
        Import(line_number=2, indented=False, module="os", attribute="sep", file_path=file_path)
    ]

    # Test case 10: From import with alias and continuation line
    input_stream = io.StringIO("from os import path as pth, \\\n    sep as separator")
    assert list(imports(input_stream, config, file_path)) == [
        Import(line_number=1, indented=False, module="os", attribute="path", alias="pth", file_path=file_path),
        Import(line_number=2, indented=False, module="os", attribute="sep", alias="separator", file_path=file_path)
    ]


# LLM-generated content at query #14
#--------------------------

# Unit test for method statement of class Import
def test_Import_statement():
    # Test basic import
    imp = Import(1, False, "os")
    assert imp.statement() == "import os"

    # Test cimport
    imp = Import(2, False, "numpy", cimport=True)
    assert imp.statement() == "cimport numpy"

    # Test from import
    imp = Import(3, False, "math", "sqrt")
    assert imp.statement() == "from math import sqrt"

    # Test from cimport
    imp = Import(4, False, "cython", "parallel", cimport=True)
    assert imp.statement() == "from cython cimport parallel"

    # Test import with alias
    imp = Import(5, False, "pandas", alias="pd")
    assert imp.statement() == "import pandas as pd"

    # Test from import with alias
    imp = Import(6, False, "numpy", "array", alias="arr")
    assert imp.statement() == "from numpy import array as arr"

    # Test from cimport with alias
    imp = Import(7, False, "cython", "parallel", alias="par", cimport=True)
    assert imp.statement() == "from cython cimport parallel as par"



# LLM-generated content at query #15
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test basic import
    input_stream = StringIO("import os\n")
    assert list(imports(input_stream)) == [Import(1, False, "os")]

    # Test from import
    input_stream = StringIO("from os import path\n")
    assert list(imports(input_stream)) == [Import(1, False, "os", "path")]

    # Test import with alias
    input_stream = StringIO("import os as operating_system\n")
    assert list(imports(input_stream)) == [Import(1, False, "os", alias="operating_system")]

    # Test from import with alias
    input_stream = StringIO("from os import path as p\n")
    assert list(imports(input_stream)) == [Import(1, False, "os", "path", alias="p")]

    # Test cimport
    input_stream = StringIO("cimport numpy as np\n")
    assert list(imports(input_stream)) == [Import(1, False, "numpy", alias="np", cimport=True)]

    # Test indented import
    input_stream = StringIO("    import os\n")
    assert list(imports(input_stream)) == [Import(1, True, "os")]

    # Test top_only flag
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    assert len(list(imports(input_stream, top_only=True))) == 1

    # Test multi-line import
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    assert list(imports(input_stream)) == [
        Import(1, False, "os", "path"),
        Import(2, False, "os", "environ"),
    ]

    # Test import with comments
    input_stream = StringIO("import os  # comment\n")
    assert list(imports(input_stream)) == [Import(1, False, "os")]

    # Test import with syntax stripping
    input_stream = StringIO("import os.path as path\n")
    assert list(imports(input_stream)) == [Import(1, False, "os.path", alias="path")]

    # Test complex import
    input_stream = StringIO("from os import path as p, environ as e\n")
    assert list(imports(input_stream)) == [
        Import(1, False, "os", "path", alias="p"),
        Import(1, False, "os", "environ", alias="e"),
    ]

    # Test import with file path
    file_path = Path("test.py")
    input_stream = StringIO("import os\n")
    assert list(imports(input_stream, file_path=file_path)) == [Import(1, False, "os", file_path=file_path)]


# LLM-generated content at query #16
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test case 1: Simple import
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 2: From import
    input_stream = StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 3: Import with alias
    input_stream = StringIO("import os as operating_system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"

    # Test case 4: Cimport
    input_stream = StringIO("cimport numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport is True

    # Test case 5: Multiline import
    input_stream = StringIO("from os import (\n    path,\n    environ\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 6: Indented import
    input_stream = StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].indented is True

    # Test case 7: File path handling
    file_path = Path("test.py")
    input_stream = StringIO("import os\n")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test case 8: Top only flag
    input_stream = StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 9: Redundant alias removal
    config = DEFAULT_CONFIG.copy()
    config.remove_redundant_aliases = True
    input_stream = StringIO("import os as os\n")
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test case 10: Complex multiline import
    input_stream = StringIO("from os import (\n    path as p,\n    environ as e,\n    sep as s\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[1].alias == "e"
    assert result[2].module == "os"
    assert result[2].attribute == "sep"
    assert result[2].alias == "s"


# LLM-generated content at query #17
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__(): 
    file_path = Path("test_file.py")
    import_instance = Import(line_number=5, indented=True, module="module_name", file_path=file_path)
    expected_output = f"{file_path}:5 indented import module_name"
    assert str(import_instance) == expected_output

    import_instance = Import(line_number=10, indented=False, module="another_module", file_path=None)
    expected_output = ":10 import another_module"
    assert str(import_instance) == expected_output

    import_instance = Import(line_number=15, indented=True, module="module", attribute="attribute", file_path=file_path)
    expected_output = f"{file_path}:15 indented from module import attribute"
    assert str(import_instance) == expected_output

    import_instance = Import(line_number=20, indented=False, module="module", attribute="attribute", alias="alias", file_path=file_path)
    expected_output = f"{file_path}:20 from module import attribute as alias"
    assert str(import_instance) == expected_output

    import_instance = Import(line_number=25, indented=True, module="module", cimport=True, file_path=file_path)
    expected_output = f"{file_path}:25 indented cimport module"
    assert str(import_instance) == expected_output

    import_instance = Import(line_number=30, indented=False, module="module", attribute="attribute", cimport=True, file_path=file_path)
    expected_output = f"{file_path}:30 from module cimport attribute"
    assert str(import_instance) == expected_output


# LLM-generated content at query #18
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import1 = Import(1, False, "module1", None, None, False, None)
    assert str(import1) == ":1 import module1"

    import2 = Import(2, True, "module2", "attribute", None, False, Path("test.py"))
    assert str(import2) == "test.py:2 indented from module2 import attribute"

    import3 = Import(3, False, "module3", None, "alias", True, Path("test.py"))
    assert str(import3) == "test.py:3 cimport module3 as alias"

    import4 = Import(4, True, "module4", "attribute", "alias", False, Path("test.py"))
    assert str(import4) == "test.py:4 indented from module4 import attribute as alias"


# LLM-generated content at query #19
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    file_path = Path("example.py")
    import_instance = Import(line_number=10, indented=True, module="os", file_path=file_path)
    expected = "example.py:10 indented import os"
    assert str(import_instance) == expected

    import_instance = Import(line_number=5, indented=False, module="sys", attribute="path", file_path=None)
    expected = ":5 from sys import path"
    assert str(import_instance) == expected

    import_instance = Import(line_number=7, indented=False, module="math", alias="m", file_path=file_path)
    expected = "example.py:7 import math as m"
    assert str(import_instance) == expected

    import_instance = Import(line_number=3, indented=True, module="numpy", cimport=True, file_path=file_path)
    expected = "example.py:3 indented cimport numpy"
    assert str(import_instance) == expected

    import_instance = Import(line_number=8, indented=False, module="pandas", attribute="DataFrame", alias="df", file_path=file_path)
    expected = "example.py:8 from pandas import DataFrame as df"
    assert str(import_instance) == expected


# LLM-generated content at query #20
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO
    input_stream = StringIO("import os\nfrom sys import path\nimport numpy as np\n")
    config = DEFAULT_CONFIG
    file_path = Path("test.py")
    result = list(imports(input_stream, config, file_path))
    expected = [
        Import(line_number=1, indented=False, module="os", file_path=file_path),
        Import(line_number=2, indented=False, module="sys", attribute="path", file_path=file_path),
        Import(line_number=3, indented=False, module="np", alias="np", file_path=file_path),
    ]
    assert result == expected


####################################################################
# TEST GENERATION BEGINS (CODAMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# LLM-generated content at query #1
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test basic import
    input_stream = StringIO("import os\nimport sys")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test from import
    input_stream = StringIO("from os import path")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test alias
    input_stream = StringIO("import os as operating_system")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].alias == "operating_system"

    # Test cimport
    input_stream = StringIO("cimport cython")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "cython"
    assert imports_list[0].cimport is True

    # Test indented import
    input_stream = StringIO("    import os")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].indented is True

    # Test top_only
    input_stream = StringIO("import os\ndef foo(): pass")
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    # Test multi-line import
    input_stream = StringIO("import os, \\\nsys")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test from import with multi-line
    input_stream = StringIO("from os import \\\npath")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test from import with parentheses
    input_stream = StringIO("from os import (path)")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test from import with multiple attributes
    input_stream = StringIO("from os import path, sep")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[1].module == "os"
    assert imports_list[1].attribute == "sep"

    # Test from import with alias
    input_stream = StringIO("from os import path as p")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"
    assert imports_list[0].alias == "p"


# LLM-generated content at query #2
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test case 1: Simple import statement
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import statement
    input_stream = StringIO("from os import path\nfrom sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "version"

    # Test case 3: Import with alias
    input_stream = StringIO("import os as operating_system\nfrom sys import version as ver")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"
    assert result[1].module == "sys"
    assert result[1].attribute == "version"
    assert result[1].alias == "ver"

    # Test case 4: Cimport statement
    input_stream = StringIO("cimport numpy as np\nfrom cython cimport parallel")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport is True
    assert result[1].module == "cython"
    assert result[1].attribute == "parallel"
    assert result[1].cimport is True

    # Test case 5: Indented import statement
    input_stream = StringIO("    import os\n    from sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].indented is True
    assert result[1].module == "sys"
    assert result[1].attribute == "version"
    assert result[1].indented is True

    # Test case 6: Import statement with parentheses
    input_stream = StringIO("from os import (path, environ)\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[2].module == "sys"

    # Test case 7: Import statement with escaped lines
    input_stream = StringIO("from os import path, \\\n environ\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"
    assert result[2].module == "sys"

    # Test case 8: Import statement with comments
    input_stream = StringIO("import os  # comment\n# comment\nfrom sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[1].attribute == "version"

    # Test case 9: Multiple imports in one line
    input_stream = StringIO("import os, sys\nfrom math import sqrt, log")
    result = list(imports(input_stream))
    assert len(result) == 4
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "math"
    assert result[2].attribute == "sqrt"
    assert result[3].module == "math"
    assert result[3].attribute == "log"

    # Test case 10: Import statement with redundant alias
    input_stream = StringIO("import os as os\nfrom sys import version as version")
    result = list(imports(input_stream, config=DEFAULT_CONFIG))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias is None
    assert result[1].module == "sys"
    assert result[1].attribute == "version"
    assert result[1].alias is None


# LLM-generated content at query #3
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import1 = Import(1, False, "module1", None, None, False, None)
    assert str(import1) == ":1 import module1"

    import2 = Import(2, True, "module2", "attribute2", None, False, Path("test.py"))
    assert str(import2) == "test.py:2 indented from module2 import attribute2"

    import3 = Import(3, False, "module3", None, "alias3", True, Path("test.py"))
    assert str(import3) == "test.py:3 cimport module3 as alias3"

    import4 = Import(4, True, "module4", "attribute4", "alias4", False, Path("test.py"))
    assert str(import4) == "test.py:4 indented from module4 import attribute4 as alias4"


# LLM-generated content at query #4
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test basic import
    input_stream = io.StringIO("import os\nimport sys")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 2
    assert imports_list[0].module == "os"
    assert imports_list[1].module == "sys"

    # Test from import
    input_stream = io.StringIO("from os import path")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].attribute == "path"

    # Test aliased import
    input_stream = io.StringIO("import numpy as np")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"

    # Test cimport
    input_stream = io.StringIO("cimport numpy as np")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "numpy"
    assert imports_list[0].alias == "np"
    assert imports_list[0].cimport is True

    # Test indented import
    input_stream = io.StringIO("    import os")
    imports_list = list(imports(input_stream))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"
    assert imports_list[0].indented is True

    # Test file_path parameter
    test_path = Path("test.py")
    input_stream = io.StringIO("import os")
    imports_list = list(imports(input_stream, file_path=test_path))
    assert len(imports_list) == 1
    assert imports_list[0].file_path == test_path

    # Test top_only parameter
    input_stream = io.StringIO("import os\ndef foo():\n    import sys")
    imports_list = list(imports(input_stream, top_only=True))
    assert len(imports_list) == 1
    assert imports_list[0].module == "os"

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #5
#--------------------------

# Unit test for method statement of class Import
def test_Import_statement():
    import_obj = Import(
        line_number=1,
        indented=False,
        module='math',
        attribute='sqrt',
        alias='square_root',
        cimport=False,
        file_path=Path('test.py')
    )

    assert import_obj.statement() == 'from math import sqrt as square_root'


# LLM-generated content at query #6
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import_instance = Import(
        line_number=42,
        indented=True,
        module="example_module",
        attribute="example_attribute",
        alias="example_alias",
        cimport=True,
        file_path=Path("/path/to/file.py"),
    )
    expected_str = "/path/to/file.py:42 indented from example_module cimport example_attribute as example_alias"
    assert str(import_instance) == expected_str


# LLM-generated content at query #7
#--------------------------

# Unit test for method statement of class Import
def test_Import_statement():
    import_instance = Import(1, False, "module", "attribute", "alias", False, Path("test.py"))
    assert import_instance.statement() == "from module import attribute as alias"


# LLM-generated content at query #8
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    # Test case 1: Basic import statement
    import1 = Import(line_number=1, indented=False, module="os")
    assert str(import1) == ":1 import os"

    # Test case 2: Import with attribute
    import2 = Import(line_number=2, indented=True, module="sys", attribute="path")
    assert str(import2) == ":2 indented from sys import path"

    # Test case 3: Import with alias
    import3 = Import(line_number=3, indented=False, module="numpy", alias="np")
    assert str(import3) == ":3 import numpy as np"

    # Test case 4: Cimport statement
    import4 = Import(line_number=4, indented=True, module="cython", cimport=True)
    assert str(import4) == ":4 indented cimport cython"

    # Test case 5: Import with file path
    import5 = Import(line_number=5, indented=False, module="pandas", file_path=Path("/test.py"))
    assert str(import5) == "/test.py:5 import pandas"

    # Test case 6: Complex import with all attributes
    import6 = Import(
        line_number=6,
        indented=True,
        module="module",
        attribute="attr",
        alias="alias",
        cimport=True,
        file_path=Path("/complex.py"),
    )
    assert str(import6) == "/complex.py:6 indented from module cimport attr as alias"


# LLM-generated content at query #9
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import tempfile
    from pathlib import Path

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"

    # Test case 4: From import with alias
    input_stream = io.StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"

    # Test case 5: Cimport
    input_stream = io.StringIO("cimport numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport is True

    # Test case 6: Indented import
    input_stream = io.StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].indented is True

    # Test case 7: File path
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"import os\n")
        temp_file_path = Path(temp_file.name)
    with open(temp_file_path, "r") as input_stream:
        result = list(imports(input_stream, file_path=temp_file_path))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].file_path == temp_file_path
    temp_file_path.unlink()

    # Test case 8: Top only
    input_stream = io.StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #10
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    from pathlib import Path

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"

    # Test case 4: From import with alias
    input_stream = io.StringIO("from os import path as p\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"

    # Test case 5: Cimport
    input_stream = io.StringIO("cimport numpy as np\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport is True

    # Test case 6: Indented import
    input_stream = io.StringIO("    import os\n")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].indented is True

    # Test case 7: Multiline import
    input_stream = io.StringIO("from os import (\n    path,\n    name\n)\n")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "name"

    # Test case 8: Top only flag
    input_stream = io.StringIO("import os\ndef foo():\n    import sys\n")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 9: File path included
    file_path = Path("test.py")
    input_stream = io.StringIO("import os\n")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #11
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test case 1: Simple import
    input_stream = StringIO("import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute is None
    assert result[0].alias is None

    # Test case 2: From import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None

    # Test case 3: Import with alias
    input_stream = StringIO("import os as operating_system")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"

    # Test case 4: From import with alias
    input_stream = StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"

    # Test case 5: Multiple imports
    input_stream = StringIO("import os, sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 6: Indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].indented

    # Test case 7: Cimport
    input_stream = StringIO("cimport cython")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport

    # Test case 8: From Cimport
    input_stream = StringIO("from cython cimport parallel")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].cimport
    assert result[0].module == "cython"
    assert result[0].attribute == "parallel"

    # Test case 9: Import with redundant alias
    input_stream = StringIO("import os as os")
    result = list(imports(input_stream, config=Config(remove_redundant_aliases=True)))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test case 10: From import with redundant alias
    input_stream = StringIO("from os import path as path")
    result = list(imports(input_stream, config=Config(remove_redundant_aliases=True)))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None


# LLM-generated content at query #12
#--------------------------

# Unit test for function imports
def test_imports():
    import io

    # Test case 1: Simple import statement
    input_stream = io.StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: Import with alias
    input_stream = io.StringIO("import os as operating_system")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"

    # Test case 3: From import statement
    input_stream = io.StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 4: From import statement with alias
    input_stream = io.StringIO("from os import path as p")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "p"

    # Test case 5: Multi-line import statement
    input_stream = io.StringIO("from os import \\\n    path,\n    environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 6: Import statement with comments
    input_stream = io.StringIO("import os  # This is a comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 7: Import statement with inline comment
    input_stream = io.StringIO("import os  # This is a comment\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 8: Import statement with multiple imports
    input_stream = io.StringIO("import os, sys, math")
    result = list(imports(input_stream))
    assert len(result) == 3
    assert result[0].module == "os"
    assert result[1].module == "sys"
    assert result[2].module == "math"

    # Test case 9: Import statement with multiple imports and aliases
    input_stream = io.StringIO("import os as operating_system, sys as system")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"
    assert result[1].module == "sys"
    assert result[1].alias == "system"

    # Test case 10: Import statement with multiple from imports
    input_stream = io.StringIO("from os import path, environ")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"


# LLM-generated content at query #13
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    from pathlib import Path

    # Test basic import
    input_stream = io.StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test from import
    input_stream = io.StringIO("from os import path\nfrom sys import version")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].attribute == "path"
    assert result[1].module == "sys" and result[1].attribute == "version"

    # Test aliased import
    input_stream = io.StringIO("import os as operating_system\nfrom sys import version as v")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os" and result[0].alias == "operating_system"
    assert result[1].module == "sys" and result[1].attribute == "version" and result[1].alias == "v"

    # Test cimport
    input_stream = io.StringIO("cimport numpy\nfrom numpy cimport array")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "numpy" and result[0].cimport
    assert result[1].module == "numpy" and result[1].attribute == "array" and result[1].cimport

    # Test indented import
    input_stream = io.StringIO("    import os\n  import sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].indented and result[0].module == "os"
    assert result[1].indented and result[1].module == "sys"

    # Test with file path
    file_path = Path("test.py")
    input_stream = io.StringIO("import os")
    result = list(imports(input_stream, file_path=file_path))
    assert len(result) == 1
    assert result[0].file_path == file_path

    # Test top_only flag
    input_stream = io.StringIO("import os\ndef foo():\n    import sys")
    result = list(imports(input_stream, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    print("All tests passed!")

if __name__ == "__main__":
    test_imports()


# LLM-generated content at query #14
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import sys
    import unittest

    class TestImports(unittest.TestCase):
        def test_imports(self):
            test_cases = [
                ("import os", [Import(1, False, "os", None, None, False, None)]),
                ("from os import path", [Import(1, False, "os", "path", None, False, None)]),
                ("import os as os_alias", [Import(1, False, "os", None, "os_alias", False, None)]),
                ("from os import path as path_alias", [Import(1, False, "os", "path", "path_alias", False, None)]),
                ("import os\nimport sys", [Import(1, False, "os", None, None, False, None), Import(2, False, "sys", None, None, False, None)]),
                ("import os\nfrom sys import path", [Import(1, False, "os", None, None, False, None), Import(2, False, "sys", "path", None, False, None)]),
                ("cimport os", [Import(1, False, "os", None, None, True, None)]),
                ("from os cimport path", [Import(1, False, "os", "path", None, True, None)]),
                ("import os\n\nimport sys", [Import(1, False, "os", None, None, False, None), Import(3, False, "sys", None, None, False, None)]),
                ("import os\n\nfrom sys import path", [Import(1, False, "os", None, None, False, None), Import(3, False, "sys", "path", None, False, None)]),
                ("import os\n\n# comment\nimport sys", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", None, None, False, None)]),
                ("import os\n\n# comment\nfrom sys import path", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", None, False, None)]),
                ("import os\n\n# comment\ncimport sys", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", None, None, True, None)]),
                ("import os\n\n# comment\nfrom sys cimport path", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", None, True, None)]),
                ("import os\n\n# comment\nfrom sys cimport path as path_alias", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", True, None)]),
                ("import os\n\n# comment\nfrom sys import path as path_alias", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", False, None)]),
                ("import os\n\n# comment\nfrom sys import path as path_alias\nimport sys", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", False, None), Import(5, False, "sys", None, None, False, None)]),
                ("import os\n\n# comment\nfrom sys import path as path_alias\ncimport sys", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", False, None), Import(5, False, "sys", None, None, True, None)]),
                ("import os\n\n# comment\nfrom sys import path as path_alias\nfrom sys cimport path", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", False, None), Import(5, False, "sys", "path", None, True, None)]),
                ("import os\n\n# comment\nfrom sys import path as path_alias\nfrom sys cimport path as path_alias", [Import(1, False, "os", None, None, False, None), Import(4, False, "sys", "path", "path_alias", False, None), Import(5, False, "sys", "path", "path_alias", True, None)]),
            ]

            for case, expected in test_cases:
                with self.subTest(case=case):
                    input_stream = io.StringIO(case)
                    result = list(imports(input_stream))
                    self.assertEqual(result, expected)

    if __name__ == "__main__":
        unittest.main()


# LLM-generated content at query #15
#--------------------------

# Unit test for method __str__ of class Import
def test_Import___str__():
    import_obj = Import(42, True, 'my_module', 'my_attr', 'my_alias', True, Path('my_file.py'))
    assert str(import_obj) == "my_file.py:42 indented from my_module cimport my_attr as my_alias"

    import_obj = Import(1, False, 'another_module', None, None, False, None)
    assert str(import_obj) == ":1 import another_module"

    import_obj = Import(99, True, 'some.module', 'some_attr', None, False, Path('another_file.py'))
    assert str(import_obj) == "another_file.py:99 indented from some.module import some_attr"

    import_obj = Import(10, False, 'yet_another.module', None, 'alias', True, None)
    assert str(import_obj) == ":10 cimport yet_another.module as alias"

    import_obj = Import(5, False, 'simple_module', None, None, False, Path('simple_file.py'))
    assert str(import_obj) == "simple_file.py:5 import simple_module"


# LLM-generated content at query #16
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    import unittest
    from pathlib import Path

    class TestImports(unittest.TestCase):
        def test_simple_import(self):
            input_stream = io.StringIO("import os\nimport sys")
            result = list(imports(input_stream))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].module, "os")
            self.assertEqual(result[1].module, "sys")

        def test_from_import(self):
            input_stream = io.StringIO("from os import path\nfrom sys import version")
            result = list(imports(input_stream))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].module, "os")
            self.assertEqual(result[0].attribute, "path")
            self.assertEqual(result[1].module, "sys")
            self.assertEqual(result[1].attribute, "version")

        def test_import_with_alias(self):
            input_stream = io.StringIO("import os as operating_system\nfrom sys import version as v")
            result = list(imports(input_stream))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].module, "os")
            self.assertEqual(result[0].alias, "operating_system")
            self.assertEqual(result[1].module, "sys")
            self.assertEqual(result[1].attribute, "version")
            self.assertEqual(result[1].alias, "v")

        def test_cimport(self):
            input_stream = io.StringIO("cimport numpy\nfrom numpy cimport array")
            result = list(imports(input_stream))
            self.assertEqual(len(result), 2)
            self.assertTrue(result[0].cimport)
            self.assertEqual(result[0].module, "numpy")
            self.assertTrue(result[1].cimport)
            self.assertEqual(result[1].module, "numpy")
            self.assertEqual(result[1].attribute, "array")

        def test_indented_import(self):
            input_stream = io.StringIO("    import os\n    from sys import version")
            result = list(imports(input_stream))
            self.assertEqual(len(result), 2)
            self.assertTrue(result[0].indented)
            self.assertTrue(result[1].indented)

        def test_top_only(self):
            input_stream = io.StringIO("import os\ndef foo():\n    import sys")
            result = list(imports(input_stream, top_only=True))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].module, "os")

        def test_file_path(self):
            input_stream = io.StringIO("import os")
            file_path = Path("/test/path.py")
            result = list(imports(input_stream, file_path=file_path))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].file_path, file_path)

    unittest.main()


# LLM-generated content at query #17
#--------------------------

# Unit test for function imports
def test_imports():
    from io import StringIO

    # Test case 1: Simple import
    input_stream = StringIO("import os\nimport sys")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = StringIO("from os import path")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 3: Aliased import
    input_stream = StringIO("import numpy as np")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "numpy"
    assert result[0].alias == "np"

    # Test case 4: Cimport
    input_stream = StringIO("cimport cython")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "cython"
    assert result[0].cimport

    # Test case 5: Multiline import
    input_stream = StringIO("from os import (\n    path,\n    environ\n)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 6: Indented import
    input_stream = StringIO("    import os")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].indented

    # Test case 7: Import with redundant alias
    input_stream = StringIO("import os as os")
    config = DEFAULT_CONFIG._replace(remove_redundant_aliases=True)
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].alias is None

    # Test case 8: From import with redundant alias
    input_stream = StringIO("from os import path as path")
    config = DEFAULT_CONFIG._replace(remove_redundant_aliases=True)
    result = list(imports(input_stream, config=config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias is None

    # Test case 9: Import with continuation lines
    input_stream = StringIO("from os import (\\\n    path,\n    environ)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 10: Import with comment
    input_stream = StringIO("import os # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 11: Import with comment and continuation lines
    input_stream = StringIO("from os import (\\\n    path, # comment\n    environ)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 12: Import with syntax characters
    input_stream = StringIO("from os import path as {|path|}")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[0].alias == "path"

    # Test case 13: Import with syntax characters and continuation lines
    input_stream = StringIO("from os import (\\\n    {|path|},\n    environ)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"

    # Test case 14: Import with syntax characters and comment
    input_stream = StringIO("from os import {|path|} # comment")
    result = list(imports(input_stream))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].attribute == "path"

    # Test case 15: Import with syntax characters and continuation lines and comment
    input_stream = StringIO("from os import (\\\n    {|path|}, # comment\n    environ)")
    result = list(imports(input_stream))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "os"
    assert result[1].attribute == "environ"


# LLM-generated content at query #18
#--------------------------

# Unit test for function imports
def test_imports():
    import io
    from pathlib import Path

    config = Config()

    # Test case 1: Simple import
    input_stream = io.StringIO("import os\nimport sys")
    result = list(imports(input_stream, config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[1].module == "sys"

    # Test case 2: From import
    input_stream = io.StringIO("from os import path\nfrom sys import version")
    result = list(imports(input_stream, config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].attribute == "path"
    assert result[1].module == "sys"
    assert result[1].attribute == "version"

    # Test case 3: Import with alias
    input_stream = io.StringIO("import os as operating_system\nfrom sys import version as ver")
    result = list(imports(input_stream, config))
    assert len(result) == 2
    assert result[0].module == "os"
    assert result[0].alias == "operating_system"
    assert result[1].module == "sys"
    assert result[1].attribute == "version"
    assert result[1].alias == "ver"

    # Test case 4: Cimport
    input_stream = io.StringIO("cimport numpy as np\nfrom cython cimport boundscheck")
    result = list(imports(input_stream, config))
    assert len(result) == 2
    assert result[0].module == "numpy"
    assert result[0].alias == "np"
    assert result[0].cimport == True
    assert result[1].module == "cython"
    assert result[1].attribute == "boundscheck"
    assert result[1].cimport == True

    # Test case 5: Indented import
    input_stream = io.StringIO("def func():\n\timport os")
    result = list(imports(input_stream, config))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].indented == True

    # Test case 6: Top only
    input_stream = io.StringIO("import os\ndef func():\n\timport sys")
    result = list(imports(input_stream, config, top_only=True))
    assert len(result) == 1
    assert result[0].module == "os"

    # Test case 7: File path
    file_path = Path("test.py")
    input_stream = io.StringIO("import os")
    result = list(imports(input_stream, config, file_path=file_path))
    assert len(result) == 1
    assert result[0].module == "os"
    assert result[0].file_path == file_path

    print("All tests passed!")

test_imports()


# LLM-generated content at query #19
#--------------------------

# Unit test for function imports
def test_imports():  # pragma: no cover
    """Test the imports function."""
    import io

    test_input = io.StringIO(
        '''import os
from sys import path
from collections import defaultdict as dd
'''
    )
    expected = [
        Import(1, False, "os"),
        Import(2, False, "sys", "path"),
        Import(3, False, "collections", "defaultdict", "dd"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path
from sys import path as sys_path
'''
    )
    expected = [
        Import(1, False, "os.path"),
        Import(2, False, "sys", "path", "sys_path"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
        Import(6, False, "...", "grandparent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
from .... import great_grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
        Import(6, False, "...", "grandparent_relative"),
        Import(7, False, "....", "great_grandparent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
from .... import great_grandparent_relative
from ..... import great_great_grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
        Import(6, False, "...", "grandparent_relative"),
        Import(7, False, "....", "great_grandparent_relative"),
        Import(8, False, ".....", "great_great_grandparent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
from .... import great_grandparent_relative
from ..... import great_great_grandparent_relative
from ...... import great_great_great_grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
        Import(6, False, "...", "grandparent_relative"),
        Import(7, False, "....", "great_grandparent_relative"),
        Import(8, False, ".....", "great_great_grandparent_relative"),
        Import(9, False, "......", "great_great_great_grandparent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
from .... import great_grandparent_relative
from ..... import great_great_grandparent_relative
from ...... import great_great_great_grandparent_relative
from ....... import great_great_great_great_grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".", "relative"),
        Import(5, False, "..", "parent_relative"),
        Import(6, False, "...", "grandparent_relative"),
        Import(7, False, "....", "great_grandparent_relative"),
        Import(8, False, ".....", "great_great_grandparent_relative"),
        Import(9, False, "......", "great_great_great_grandparent_relative"),
        Import(10, False, ".......", "great_great_great_great_grandparent_relative"),
    ]
    assert list(imports(test_input)) == expected

    test_input = io.StringIO(
        '''import os.path as ospath
from sys import path as sys_path
from collections import defaultdict as dd
from . import relative
from .. import parent_relative
from ... import grandparent_relative
from .... import great_grandparent_relative
from ..... import great_great_grandparent_relative
from ...... import great_great_great_grandparent_relative
from ....... import great_great_great_great_grandparent_relative
from ........ import great_great_great_great_great_grandparent_relative
'''
    )
    expected = [
        Import(1, False, "os.path", None, "ospath"),
        Import(2, False, "sys", "path", "sys_path"),
        Import(3, False, "collections", "defaultdict", "dd"),
        Import(4, False, ".",


