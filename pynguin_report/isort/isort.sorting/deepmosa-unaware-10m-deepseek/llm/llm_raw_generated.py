####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import statements
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test 2: Force to top functionality
    config.force_to_top = {"os", "sys"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("from sys import path", config) == "Afrom sys import path"
    assert section_key("import datetime", config) == "Bimport datetime"
    
    # Test 3: Case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from SYS import path", config) == "Bfrom sys import path"
    
    # Test 4: Length sort
    config.length_sort = True
    config.case_sensitive = True
    config.order_by_type = True
    short_line = "import a"
    long_line = "import abcdef"
    assert section_key(short_line, config) == f"B{len(short_line)}{short_line}"
    assert section_key(long_line, config) == f"B{len(long_line)}{long_line}"
    
    # Test 5: Relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    config.length_sort = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 6: Group by package
    config.group_by_package = True
    config.reverse_relative = False
    assert section_key("from package import module", config) == "Bfrom package"
    assert section_key("from package.sub import thing", config) == "Bfrom package.sub"
    
    # Test 7: Lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    
    # Test 8: Sort relative in force sorted sections
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    assert section_key("from . import mod", config) == "Bfrom . import mod"
    config.reverse_relative = False
    assert section_key("from . import mod", config) == "Bfrom ._ import mod"
    
    # Test 9: Honor case in force sorted sections
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("from MODULE import Class", config) == "Bfrom MODULE import class"
    
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("from MODULE import Class", config) == "Bfrom module import Class"
    
    # Test 10: Complex mixed case scenario
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    line = "from MyModule import MyClass, my_function"
    result = section_key(line, config)
    assert "MyModule" in result
    assert "myclass, my_function" in result.lower()
    
    # Test 11: Edge cases
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    assert section_key("", config) == "B"
    assert section_key("import", config) == "Bimport"
    
    # Test 12: Multiple force to top entries
    config.force_to_top = {"zebra", "apple"}
    assert section_key("import zebra", config) == "Aimport zebra"
    assert section_key("import apple", config) == "Aimport apple"
    assert section_key("import banana", config) == "Bimport banana"


# LLM-generated content at query #2
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic functionality with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bos"
    assert section_key("from os import path", config) == "Bpath"
    
    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aos"
    assert section_key("from os import path", config) == "Apath"
    
    # Test case sensitivity
    config.case_sensitive = False
    config.force_to_top = set()
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    
    # Test length_sort
    config.length_sort = True
    config.case_sensitive = True
    assert section_key("import a", config) == "B1a"
    assert section_key("import abc", config) == "B3abc"
    
    # Test reverse_relative with from . imports
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    config.length_sort = False
    assert section_key("from . import module", config) == "B. import module"
    
    # Test group_by_package
    config.reverse_relative = False
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bpackage"
    
    # Test lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    
    # Test sort_relative_in_force_sorted_sections
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    assert section_key("from .. import module", config) == "B.. module"
    
    # Test honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("import OS", config) == "BOS"
    assert section_key("from OS import PATH", config) == "BOS import PATH"
    
    # Test with order_by_type=False
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    config.case_sensitive = True
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    
    # Test complex import line
    config.order_by_type = True
    config.case_sensitive = True
    config.force_to_top = {"django"}
    assert section_key("from django.contrib import admin", config) == "Aadmin"


# LLM-generated content at query #3
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = set()
    
    assert module_key("module", config) == "Bmodule"
    
    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    assert module_key("..module", config) == "B.. module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    assert module_key("..module", config) == "B.._module"
    
    # Test ignore_case parameter
    assert module_key("Module", config, ignore_case=True) == "Bmodule"
    assert module_key("Module", config, ignore_case=False) == "BModule"
    
    # Test case_sensitive config
    config.case_sensitive = False
    assert module_key("Module", config) == "bmodule"
    
    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}
    
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    
    # Test uppercase module name detection (issue #376)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    
    # Test class detection by first character
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    
    # Test default prefix for other modules
    assert module_key("other", config, sub_imports=True) == "BCother"
    
    # Test length_sort
    config.length_sort = True
    assert module_key("module", config) == "B3:module"
    assert module_key("longmodule", config) == "B10:longmodule"
    
    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["test_section"]
    assert module_key("module", config, section_name="test_section") == "B6:module"
    assert module_key("module", config, section_name="other_section") == "Bmodule"
    
    # Test force_to_top
    config.force_to_top = {"special"}
    assert module_key("special", config) == "Aspecial"
    assert module_key("regular", config) == "Bregular"
    
    # Test combined scenarios
    config.reverse_relative = False
    config.order_by_type = True
    config.case_sensitive = True
    config.length_sort = True
    config.force_to_top = {"important"}
    
    result = module_key("..important", config, sub_imports=True)
    assert result == "A.. important" or result == "A2:.. important"


# LLM-generated content at query #4
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("module", config) == "Bmodule"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("Module", config) == "Bmodule"
    config.case_sensitive = True

    # Test force_to_top
    config.force_to_top = {"module"}
    assert module_key("module", config) == "Amodule"
    config.force_to_top = set()

    # Test relative imports
    config.reverse_relative = False
    assert module_key("..module", config) == "B.._module"
    
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}

    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    assert module_key("Other", config, sub_imports=True) == "BBOther"
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test uppercase module name (issue #376)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"

    # Test length_sort
    config.length_sort = True
    assert module_key("module", config) == "B6:module"
    assert module_key("longmodule", config) == "B10:longmodule"

    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"test"}
    assert module_key("module", config, section_name="test") == "B6:module"
    assert module_key("module", config, section_name="other") == "Bmodule"

    # Test ignore_case parameter
    assert module_key("Module", config, ignore_case=True) == "Bmodule"
    assert module_key("Module", config, ignore_case=False) == "BModule"

    # Test combined scenarios
    config.order_by_type = True
    config.force_to_top = {"Class"}
    config.length_sort = True
    result = module_key("Class", config, sub_imports=True)
    assert result.startswith("AB")  # A from force_to_top, B from order_by_type
    assert "Class" in result


# LLM-generated content at query #5
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = set()
    
    result = module_key("module", config)
    assert result == "Bmodule"
    
    # Test relative imports with reverse_relative=False
    result = module_key("..module", config)
    assert result == "B.. module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"
    
    # Test ignore_case parameter
    config.reverse_relative = False
    result = module_key("MODULE", config, ignore_case=True)
    assert result == "Bmodule"
    
    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("MODULE", config)
    assert result == "Bmodule"
    
    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.case_sensitive = True
    
    # Test constant
    config.constants = {"CONSTANT"}
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"
    
    # Test class
    config.classes = {"MyClass"}
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"
    
    # Test variable
    config.variables = {"my_var"}
    result = module_key("my_var", config, sub_imports=True)
    assert result == "BCmy_var"
    
    # Test uppercase module (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"
    
    # Test class-like module (starts with uppercase)
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"
    
    # Test default prefix for other modules
    result = module_key("regular_module", config, sub_imports=True)
    assert result == "BCregular_module"
    
    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"
    
    # Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["test_section"]
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"
    
    # Test force_to_top
    config.force_to_top = {"special_module"}
    result = module_key("special_module", config)
    assert result == "Aspecial_module"
    
    # Test combined: force_to_top with length_sort
    config.length_sort = True
    result = module_key("special_module", config)
    assert result == "A14:special_module"
    
    # Test combined: order_by_type with length_sort
    config.order_by_type = True
    config.length_sort = True
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BB7:MyClass"
    
    # Test empty module name
    result = module_key("", config)
    assert result == "B0:"
    
    # Test module name with numbers
    result = module_key("module123", config)
    assert result == "B10:module123"
    
    # Test with None section_name
    config.length_sort_sections = ["none"]
    result = module_key("module", config, section_name=None)
    assert result == "Bmodule"


# LLM-generated content at query #6
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("os", config) == "Bos"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"
    config.case_sensitive = True

    # Test force_to_top
    config.force_to_top = {"os"}
    assert module_key("os", config) == "Aos"
    config.force_to_top = set()

    # Test relative imports
    config.reverse_relative = False
    assert module_key("..module", config) == "B.._module"
    
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"ClassName"}
    config.variables = {"variable"}
    
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    
    # Test uppercase module name > 1 char
    assert module_key("ABC", config, sub_imports=True) == "BAABC"
    
    # Test class detection by first uppercase letter
    assert module_key("MyClass", config, sub_imports=True) == "BBMyClass"
    
    # Test default prefix for other modules
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    assert module_key("long_module", config) == "B11:long_module"
    assert module_key("short", config) == "B5:short"

    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"future"}
    assert module_key("module", config, section_name="future") == "B6:module"
    assert module_key("module", config, section_name="standard") == "Bmodule"

    # Test ignore_case parameter
    assert module_key("MODULE", config, ignore_case=True) == "Bmodule"
    assert module_key("Module", config, ignore_case=False) == "BModule"

    # Test combined scenarios
    config.force_to_top = {"important"}
    config.length_sort = True
    config.order_by_type = True
    config.constants = {"VALUE"}
    
    assert module_key("important", config, sub_imports=True) == "A6:important"
    assert module_key("VALUE", config, sub_imports=True) == "BA5:VALUE"


# LLM-generated content at query #7
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic section assignment
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top={"django", "requests"},
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Force to top module should get section A
    assert section_key("import django", config) == "Aimport django"
    assert section_key("from django import something", config) == "Afrom django import something"
    
    # Regular module should get section B
    assert section_key("import numpy", config) == "Bimport numpy"
    
    # Test 2: Reverse relative handling
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    
    line = "from .. import module"
    result = section_key(line, config)
    assert result.startswith("Bfrom .. import module")
    
    # Test 3: Group by package
    config.group_by_package = True
    config.reverse_relative = False
    
    line = "from package.subpackage import module"
    result = section_key(line, config)
    assert result == "Bfrom package.subpackage"
    
    # Test 4: Lexicographical mode
    config.group_by_package = False
    config.lexicographical = True
    
    line = "from package import module"
    result = section_key(line, config)
    assert result == "Bpackage.module"
    
    line = "import package.module"
    result = section_key(line, config)
    assert result == "Bpackage.module"
    
    # Test 5: Non-lexicographical mode
    config.lexicographical = False
    
    line = "from package import module"
    result = section_key(line, config)
    assert result == "Bpackage import module"
    
    line = "import package.module"
    result = section_key(line, config)
    assert result == "Bpackage.module"
    
    # Test 6: Sort relative in force sorted sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    
    line = "from . import module"
    result = section_key(line, config)
    assert result == "Bfrom . import module"
    
    config.reverse_relative = False
    line = "from .. import module"
    result = section_key(line, config)
    assert result == "Bfrom .._ import module"
    
    # Test 7: Case handling with honor_case_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    
    line = "from MyPackage import MyClass"
    result = section_key(line, config)
    assert result == "Bfrom mypackage import MyClass"
    
    config.order_by_type = False
    line = "from MyPackage import MyClass"
    result = section_key(line, config)
    assert result == "Bfrom mypackage import myclass"
    
    # Test 8: Case handling without honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    
    line = "from MyPackage import MyClass"
    result = section_key(line, config)
    assert result == "Bfrom mypackage import myclass"
    
    # Test 9: Length sort
    config.length_sort = True
    config.order_by_type = True
    
    line = "import short"
    result = section_key(line, config)
    assert result == "B5import short"
    
    line = "import verylongname"
    result = section_key(line, config)
    assert result == "B13import verylongname"
    
    # Test 10: Import statement variations
    config.length_sort = False
    config.lexicographical = True
    
    line = "import module"
    result = section_key(line, config)
    assert result == "Bmodule"
    
    line = "from module import function"
    result = section_key(line, config)
    assert result == "Bmodule.function"
    
    # Test 11: Force to top with different import styles
    config.lexicographical = False
    config.force_to_top = {"os", "sys"}
    
    line = "import os"
    result = section_key(line, config)
    assert result == "Aimport os"
    
    line = "from os import path"
    result = section_key(line, config)
    assert result == "Afrom os import path"
    
    # Test 12: Empty or whitespace lines
    line = ""
    result = section_key(line, config)
    assert result == "B"
    
    line = "   "
    result = section_key(line, config)
    assert result == "B   "


# LLM-generated content at query #8
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without special handling
    config = Mock(
        reverse_relative=False,
        order_by_type=False,
        constants=set(),
        classes=set(),
        variables=set(),
        case_sensitive=True,
        length_sort=False,
        length_sort_straight=False,
        length_sort_sections=set(),
        force_to_top=set(),
    )
    assert module_key("module", config) == "Bmodule"

    # Test with relative imports
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"
    config.reverse_relative = False
    assert module_key("..module", config) == "B.._module"

    # Test ignore_case parameter
    assert module_key("MODULE", config, ignore_case=True) == "Bmodule"
    assert module_key("Module", config, ignore_case=True) == "Bmodule"

    # Test case_sensitive config
    config.case_sensitive = False
    assert module_key("MODULE", config) == "Bmodule"
    config.case_sensitive = True

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}

    # Test constants
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    # Test uppercase modules (issue #376)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    # Test classes
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    # Test variables
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    # Test default (no match)
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test length_sort
    config.length_sort = True
    assert module_key("module", config) == "B6:module"
    assert module_key("longmodule", config) == "B10:longmodule"

    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"future"}
    assert module_key("module", config, section_name="future") == "B6:module"
    assert module_key("module", config, section_name="other") == "Bmodule"

    # Test force_to_top
    config.force_to_top = {"special"}
    assert module_key("special", config) == "Aspecial"
    assert module_key("regular", config) == "Bregular"

    # Test combination of multiple features
    config.order_by_type = True
    config.length_sort = True
    config.force_to_top = {"special"}
    result = module_key("special", config, sub_imports=True)
    assert result.startswith("A")  # Force to top takes precedence
    result = module_key("Class", config, sub_imports=True)
    assert result.startswith("BB") and ":" in result  # Has prefix and length sort


# LLM-generated content at query #9
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic import line with default config
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False
    
    line = "import os"
    result = section_key(line, config)
    assert result == "Bimport os"
    
    # Test 2: Force to top module
    config.force_to_top = ["os"]
    result = section_key(line, config)
    assert result == "Aimport os"
    
    # Test 3: From import with group_by_package
    config.group_by_package = True
    config.force_to_top = []
    line = "from collections import defaultdict"
    result = section_key(line, config)
    assert result == "Bfrom collections"
    
    # Test 4: Lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    line = "from collections import defaultdict"
    result = section_key(line, config)
    assert result == "Bcollections.defaultdict"
    
    # Test 5: Case insensitive with order_by_type=False
    config.lexicographical = False
    config.order_by_type = False
    config.case_sensitive = False
    line = "import OS"
    result = section_key(line, config)
    assert result == "Bimport os"
    
    # Test 6: Relative import with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    line = "from . import module"
    result = section_key(line, config)
    assert result == "Bfrom . import module"
    
    # Test 7: Relative import with sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    line = "from .. import module"
    result = section_key(line, config)
    assert result == "Bfrom .. import module"
    
    # Test 8: Length sort enabled
    config.length_sort = True
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    line = "import os"
    result = section_key(line, config)
    assert result == "B7import os"
    
    # Test 9: Honor case in force sorted sections with case_sensitive != order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    line = "from Collections import DefaultDict"
    result = section_key(line, config)
    assert result == "Bfrom Collections import defaultdict"
    
    # Test 10: Complex case with multiple conditions
    config.force_to_top = ["sys"]
    config.group_by_package = True
    config.lexicographical = True
    config.honor_case_in_force_sorted_sections = False
    line = "from sys import path"
    result = section_key(line, config)
    assert result == "Asys"
    
    # Test 11: Import statement with lexicographical
    config.force_to_top = []
    line = "import os.path"
    result = section_key(line, config)
    assert result == "Bos.path"
    
    # Test 12: Empty line
    line = ""
    result = section_key(line, config)
    assert result == "B"


# LLM-generated content at query #10
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(
            self,
            reverse_relative=False,
            order_by_type=False,
            constants=None,
            classes=None,
            variables=None,
            case_sensitive=True,
            length_sort=False,
            length_sort_straight=False,
            length_sort_sections=None,
            force_to_top=None,
        ):
            self.reverse_relative = reverse_relative
            self.order_by_type = order_by_type
            self.constants = constants or set()
            self.classes = classes or set()
            self.variables = variables or set()
            self.case_sensitive = case_sensitive
            self.length_sort = length_sort
            self.length_sort_straight = length_sort_straight
            self.length_sort_sections = length_sort_sections or set()
            self.force_to_top = force_to_top or set()

    config = MockConfig()
    
    # Test basic module name
    assert module_key("module", config) == "Bmodule"
    
    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    assert module_key(".module", config) == "B_module"
    assert module_key("..module", config) == "B__module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    assert module_key(".module", config) == "B .module"
    assert module_key("..module", config) == "B ..module"
    
    # Test case sensitivity
    config.case_sensitive = True
    assert module_key("Module", config) == "BModule"
    assert module_key("module", config) == "Bmodule"
    
    config.case_sensitive = False
    assert module_key("Module", config) == "bmodule"
    assert module_key("module", config) == "bmodule"
    
    # Test ignore_case parameter
    config.case_sensitive = True
    assert module_key("Module", config, ignore_case=True) == "bmodule"
    assert module_key("module", config, ignore_case=True) == "bmodule"
    
    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}
    
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    
    # Test uppercase module name detection (issue #376)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    assert module_key("A", config, sub_imports=True) == "BBCA"  # Single char not treated as constant
    
    # Test class detection by first uppercase letter
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    assert module_key("className", config, sub_imports=True) == "BCclassName"
    
    # Test length_sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"
    assert module_key("abcd", config) == "B4:abcd"
    
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("abc", config, straight_import=True) == "B3:abc"
    assert module_key("abc", config, straight_import=False) == "Babc"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"future"}
    assert module_key("abc", config, section_name="future") == "B3:abc"
    assert module_key("abc", config, section_name="standard") == "Babc"
    
    # Test force_to_top
    config.force_to_top = {"top_module"}
    assert module_key("top_module", config) == "Atop_module"
    assert module_key("other_module", config) == "Bother_module"
    
    # Test combined scenarios
    config.case_sensitive = False
    config.length_sort = True
    config.force_to_top = {"important"}
    config.constants = {"MY_CONST"}
    config.order_by_type = True
    
    result = module_key("MY_CONST", config, sub_imports=True)
    assert result == "Aamy_const" or result == "Aamy_const"  # force_to_top takes precedence
    
    result = module_key("MyClass", config, sub_imports=True)
    assert result.startswith("B")  # Not in force_to_top
    assert "myclass" in result.lower()  # case_sensitive=False


# LLM-generated content at query #11
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without config
    config = Mock(
        reverse_relative=False,
        order_by_type=False,
        constants=set(),
        classes=set(),
        variables=set(),
        case_sensitive=True,
        length_sort=False,
        length_sort_straight=False,
        length_sort_sections=set(),
        force_to_top=set(),
    )
    assert module_key("module", config) == "Bmodule"

    # Test with relative imports
    config.reverse_relative = True
    assert module_key(".module", config) == "B. module"
    config.reverse_relative = False
    assert module_key(".module", config) == "B_module"

    # Test with multiple dots
    assert module_key("...module", config) == "B___module"
    config.reverse_relative = True
    assert module_key("...module", config) == "B... module"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("Module", config) == "Bmodule"
    config.case_sensitive = True
    assert module_key("Module", config) == "BModule"

    # Test ignore_case parameter
    assert module_key("Module", config, ignore_case=True) == "Bmodule"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}

    # Test constants
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    # Test uppercase module name > 1 char (issue #376)
    assert module_key("ABC", config, sub_imports=True) == "BAABC"
    # Test classes
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    # Test variables
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    # Test other (defaults to C)
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test uppercase first character
    assert module_key("OtherClass", config, sub_imports=True) == "BBOtherClass"

    # Test length_sort
    config.length_sort = True
    assert module_key("module", config) == "B6:module"
    assert module_key("longmodule", config) == "B10:longmodule"

    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"special"}
    assert module_key("module", config, section_name="special") == "B6:module"
    assert module_key("module", config, section_name="other") == "Bmodule"

    # Test force_to_top
    config.force_to_top = {"topmodule"}
    assert module_key("topmodule", config) == "Atopmodule"
    assert module_key("othermodule", config) == "Bothermodule"

    # Test combined features
    config.length_sort = True
    config.order_by_type = True
    config.force_to_top = {"CONSTANT"}
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("AA") and "8:CONSTANT" in result


# LLM-generated content at query #12
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import statements
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("import os", config) == "Bimport os"
    
    # Test order_by_type=False
    config.order_by_type = False
    config.case_sensitive = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("import os", config) == "Bimport os"
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.order_by_type = True
    config.case_sensitive = True
    assert section_key("import os.path", config) == "Bimport os.path"
    assert section_key("from os import path", config) == "Bos.path"
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from os import path", config) == "Bfrom os"
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    
    # Test length_sort
    config.length_sort = True
    config.reverse_relative = False
    config.group_by_package = False
    assert section_key("import a", config) == "B7import a"
    assert section_key("import abc", config) == "B9import abc"
    
    # Test honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from MODULE import Class", config) == "Bfrom module import class"
    
    # Test honor_case_in_force_sorted_sections with from...import
    config.case_sensitive = False
    config.order_by_type = True
    line = "from MODULE import Class"
    result = section_key(line, config)
    assert "module" in result.lower()
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    assert section_key("from .. import module", config) == "Bfrom .._ import module"
    
    # Test complex scenario with multiple conditions
    config = Mock(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=True,
        group_by_package=True,
        lexicographical=True,
        force_to_top={"django"},
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True,
        length_sort=True
    )
    
    # This should test the interaction of multiple config options
    result = section_key("from django.conf import settings", config)
    assert result.startswith("A")  # django is in force_to_top
    
    # Test empty line
    config.force_to_top = set()
    config.length_sort = False
    assert section_key("", config) == "B"


# LLM-generated content at query #13
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock
    
    # Test basic functionality with default config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()
    
    result = module_key("module_name", config)
    assert result == "Bmodule_name"
    
    # Test with relative imports
    config.reverse_relative = False
    result = module_key("...module", config)
    assert result == "B..._module"
    
    config.reverse_relative = True
    result = module_key("...module", config)
    assert result == "B... module"
    
    # Test ignore_case parameter
    result = module_key("ModuleName", config, ignore_case=True)
    assert result == "Bmodulename"
    
    # Test case_sensitive config
    config.case_sensitive = False
    result = module_key("ModuleName", config)
    assert result == "Bmodulename"
    
    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"ClassName"}
    config.variables = {"variable_name"}
    
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"
    
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"
    
    result = module_key("variable_name", config, sub_imports=True)
    assert result == "BCvariable_name"
    
    # Test uppercase module name (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"
    
    # Test class detection by first letter
    result = module_key("SomeClass", config, sub_imports=True)
    assert result == "BBSomeClass"
    
    # Test default prefix for other modules
    result = module_key("regular_module", config, sub_imports=True)
    assert result == "BCregular_module"
    
    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"
    
    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"
    
    # Test length_sort_sections
    config.length_sort = False
    config.length_sort_straight = False
    result = module_key("module", config, section_name="test_section")
    assert result == "Bmodule"
    
    config.length_sort_sections = {"test_section"}
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"
    
    # Test force_to_top
    config.force_to_top = {"special_module"}
    result = module_key("special_module", config)
    assert result == "Aspecial_module"
    
    result = module_key("other_module", config)
    assert result == "Bother_module"
    
    # Test combined scenarios
    config.order_by_type = True
    config.length_sort = True
    config.force_to_top = {"ClassName"}
    
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "ABB7:ClassName"
    
    result = module_key("variable_name", config, sub_imports=True)
    assert result == "BBC13:variable_name"


# LLM-generated content at query #14
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock
    
    # Test basic module name handling
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()
    
    result = module_key("module", config)
    assert result == "Bmodule"
    
    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    result = module_key("..module", config)
    assert result == "B.. module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"
    
    # Test ignore_case parameter
    result = module_key("MODULE", config, ignore_case=True)
    assert result == "Bmodule"
    
    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("MODULE", config)
    assert result == "Bmodule"
    
    # Test order_by_type with constants
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"
    
    # Test order_by_type with classes
    config.constants = set()
    config.classes = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BBmodule"
    
    # Test order_by_type with variables
    config.classes = set()
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"
    
    # Test order_by_type with uppercase module (issue #376)
    config.variables = set()
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"
    
    # Test order_by_type with capitalized module
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"
    
    # Test order_by_type with lowercase module (default)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"
    
    # Test length_sort
    config.order_by_type = False
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"
    
    # Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    result = module_key("module", config, section_name="test")
    assert result == "Bmodule"
    
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="test")
    assert result == "B6:module"
    
    # Test force_to_top
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"
    
    # Test combined prefix and length sort
    config.order_by_type = True
    config.length_sort = True
    config.classes = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BB6:module"


# LLM-generated content at query #15
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic sorting with default config
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False

    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"

    # Test force_to_top
    config.force_to_top = ["sys"]
    assert section_key("import sys", config) == "Aimport sys"
    assert section_key("import os", config) == "Bimport os"

    # Test case sensitivity
    config.force_to_top = []
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from SYS import path", config) == "Bfrom sys import path"

    # Test lexicographical mode
    config.lexicographical = True
    config.case_sensitive = True
    assert section_key("import os.path", config) == "Bos.path"
    assert section_key("from os import path", config) == "Bos.path"

    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from os import path", config) == "Bfrom os"

    # Test length_sort
    config.length_sort = True
    config.group_by_package = False
    assert section_key("import a", config) == "B11import a"
    assert section_key("import abc", config) == "B13import abc"

    # Test reverse_relative handling
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"

    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"

    # Test honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport OS"
    assert section_key("from SYS import PATH", config) == "Bfrom SYS import PATH"

    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from SYS import PATH", config) == "Bfrom sys import PATH"

    # Test mixed case handling with honor_case_in_force_sorted_sections
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = True
    line = "from MixedCase import lower_name"
    result = section_key(line, config)
    assert "MixedCase" in result
    assert "lower_name" in result

    # Test empty line
    assert section_key("", config) == "B"

    # Test line with only spaces
    assert section_key("   ", config) == "B   "


# LLM-generated content at query #16
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("os", config) == "Bos"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"
    config.case_sensitive = True

    # Test force_to_top
    config.force_to_top = {"os"}
    assert module_key("os", config) == "Aos"
    config.force_to_top = set()

    # Test relative imports
    config.reverse_relative = False
    assert module_key(".os", config) == "B_os"
    config.reverse_relative = True
    assert module_key(".os", config) == "B . os"

    # Test length_sort
    config.length_sort = True
    assert module_key("os", config) == "B2:os"
    config.length_sort = False

    # Test length_sort_straight
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2:os"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"test"}
    assert module_key("os", config, section_name="test") == "B2:os"
    config.length_sort_sections = set()

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"MAX_SIZE"}
    config.classes = {"MyClass"}
    config.variables = {"my_var"}

    assert module_key("MAX_SIZE", config, sub_imports=True) == "BAmax_size"
    assert module_key("MyClass", config, sub_imports=True) == "BBmyclass"
    assert module_key("my_var", config, sub_imports=True) == "BCmy_var"
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test uppercase detection (issue #376)
    assert module_key("CONST", config, sub_imports=True) == "BAconst"

    # Test class detection by first letter
    assert module_key("ClassName", config, sub_imports=True) == "BBclassname"

    # Test ignore_case parameter
    assert module_key("MyModule", config, sub_imports=True, ignore_case=True) == "BBmymodule"

    # Test without sub_imports
    assert module_key("MyClass", config, sub_imports=False) == "Bmyclass"

    # Test combination of features
    config.force_to_top = {"important"}
    config.length_sort = True
    config.order_by_type = True
    config.constants = {"VERSION"}

    result = module_key("important", config, sub_imports=True)
    assert result.startswith("A")
    assert "important" in result

    result = module_key("VERSION", config, sub_imports=True)
    assert result.startswith("BA")
    assert "version" in result


# LLM-generated content at query #17
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(
            self,
            reverse_relative=False,
            order_by_type=False,
            constants=None,
            classes=None,
            variables=None,
            case_sensitive=True,
            length_sort=False,
            length_sort_straight=False,
            length_sort_sections=None,
            force_to_top=None,
        ):
            self.reverse_relative = reverse_relative
            self.order_by_type = order_by_type
            self.constants = constants or set()
            self.classes = classes or set()
            self.variables = variables or set()
            self.case_sensitive = case_sensitive
            self.length_sort = length_sort
            self.length_sort_straight = length_sort_straight
            self.length_sort_sections = length_sort_sections or set()
            self.force_to_top = force_to_top or set()

    # Test basic module name
    config = MockConfig()
    assert module_key("os", config) == "Bos"

    # Test relative imports with reverse_relative=False
    config = MockConfig(reverse_relative=False)
    assert module_key("..module", config) == "B_.. module"

    # Test relative imports with reverse_relative=True
    config = MockConfig(reverse_relative=True)
    assert module_key("..module", config) == "B .. module"

    # Test ignore_case parameter
    config = MockConfig()
    assert module_key("OS", config, ignore_case=True) == "Bos"
    assert module_key("Os", config, ignore_case=True) == "Bos"

    # Test case_sensitive=False
    config = MockConfig(case_sensitive=False)
    assert module_key("OS", config) == "Bos"
    assert module_key("os", config) == "Bos"

    # Test order_by_type with sub_imports
    config = MockConfig(
        order_by_type=True,
        constants={"CONSTANT"},
        classes={"ClassName"},
        variables={"variable"},
    )
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    assert module_key("Other", config, sub_imports=True) == "BBOther"

    # Test uppercase module name (issue #376)
    config = MockConfig(order_by_type=True)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"

    # Test class detection by first uppercase letter
    config = MockConfig(order_by_type=True)
    assert module_key("MyClass", config, sub_imports=True) == "BBMyClass"

    # Test default prefix for regular modules
    config = MockConfig(order_by_type=True)
    assert module_key("lowercase", config, sub_imports=True) == "BClowercase"

    # Test length_sort
    config = MockConfig(length_sort=True)
    assert module_key("os", config) == "B2:os"
    assert module_key("longmodule", config) == "B10:longmodule"

    # Test length_sort_straight with straight_import
    config = MockConfig(length_sort_straight=True)
    assert module_key("os", config, straight_import=True) == "B2:os"
    assert module_key("os", config, straight_import=False) == "Bos"

    # Test length_sort_sections
    config = MockConfig(length_sort_sections={"special"})
    assert module_key("os", config, section_name="special") == "B2:os"
    assert module_key("os", config, section_name="regular") == "Bos"

    # Test force_to_top
    config = MockConfig(force_to_top={"important"})
    assert module_key("important", config) == "Aimportant"
    assert module_key("regular", config) == "Bregular"

    # Test combined features
    config = MockConfig(
        order_by_type=True,
        constants={"MY_CONST"},
        length_sort=True,
        force_to_top={"MY_CONST"},
    )
    result = module_key("MY_CONST", config, sub_imports=True)
    assert result == "AA8:MY_CONST"

    # Test with empty module name
    config = MockConfig()
    assert module_key("", config) == "B"

    # Test numeric module names
    config = MockConfig(length_sort=True)
    assert module_key("123", config) == "B3:123"

    # Test module name with special characters
    config = MockConfig()
    assert module_key("my-module", config) == "Bmy-module"
    assert module_key("my.module", config) == "Bmy.module"


# LLM-generated content at query #18
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False,
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("import os", config) == "Bimport os"
    
    # Test length_sort
    config.length_sort = True
    config.case_sensitive = True
    assert section_key("import a", config) == "B9import a"
    assert section_key("import abc", config) == "B11import abc"
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    
    # Test group_by_package
    config.group_by_package = True
    config.reverse_relative = False
    assert section_key("from package import module", config) == "Bfrom package"
    
    # Test lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    
    # Test sort_relative_in_force_sorted_sections
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    assert section_key("from .. import module", config) == "Bfrom .._ import module"
    
    # Test honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from MODULE import Class", config) == "Bfrom MODULE import class"
    
    # Test mixed case handling
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    assert section_key("import MixedCase", config) == "Bimport mixedcase"
    
    # Test empty line
    assert section_key("", config) == "B"
    
    # Test with multiple spaces
    assert section_key("import  os", config) == "Bimport  os"
    
    # Test complex import statement
    config.order_by_type = True
    config.case_sensitive = True
    assert section_key("from package.subpackage import module as mod", config) == "Bfrom package.subpackage import module as mod"


# LLM-generated content at query #19
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic section assignment
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top={"sys", "os"},
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False
    )
    
    # Force to top modules should get section A
    assert section_key("import sys", config) == "Aimport sys"
    assert section_key("import os", config) == "Aimport os"
    
    # Other modules should get section B
    assert section_key("import math", config) == "Bimport math"
    assert section_key("from collections import defaultdict", config) == "Bfrom collections import defaultdict"
    
    # Test 2: Case sensitivity when order_by_type is False
    config.order_by_type = False
    config.case_sensitive = False
    assert section_key("import MATH", config) == "Bimport math"
    assert section_key("from COLLECTIONS import defaultdict", config) == "Bfrom collections import defaultdict"
    
    # Test 3: Case sensitivity when order_by_type is True
    config.order_by_type = True
    config.case_sensitive = False
    assert section_key("import MATH", config) == "Bimport MATH"
    assert section_key("from COLLECTIONS import defaultdict", config) == "Bfrom COLLECTIONS import defaultdict"
    
    # Test 4: honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    
    # Module name should be lowercased, names should keep case
    result = section_key("from COLLECTIONS import defaultdict, OrderedDict", config)
    assert result == "Bfrom collections import defaultdict, OrderedDict"
    
    config.case_sensitive = True
    config.order_by_type = False
    
    # Module name should keep case, names should be lowercased
    result = section_key("from collections import defaultdict, OrderedDict", config)
    assert result == "Bfrom collections import defaultdict, ordereddict"
    
    # Test 5: lexicographical sorting
    config.lexicographical = True
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    
    assert section_key("import sys", config) == "Bsys"
    assert section_key("from collections import defaultdict", config) == "Bcollections.defaultdict"
    
    # Test 6: group_by_package
    config.lexicographical = False
    config.group_by_package = True
    
    assert section_key("from collections import defaultdict", config) == "Bfrom collections"
    assert section_key("from os.path import join", config) == "Bfrom os.path"
    
    # Test 7: length_sort
    config.group_by_package = False
    config.length_sort = True
    
    result = section_key("import a", config)
    assert result.startswith("B")
    assert "7" in result  # Length of "import a"
    
    # Test 8: sort_relative_in_force_sorted_sections with reverse_relative
    config.length_sort = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 9: sort_relative_in_force_sorted_sections without reverse_relative
    config.reverse_relative = False
    
    assert section_key("from . import module", config) == "Bfrom ._ import module"
    assert section_key("from .. import module", config) == "Bfrom .._ import module"
    
    # Test 10: reverse_relative without sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = True
    
    assert section_key("from . import module", config) == "Bfrom . import module"
    
    # Test 11: Empty or whitespace lines
    config.reverse_relative = False
    config.force_to_top = set()
    
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "
    
    # Test 12: Mixed case with force_to_top
    config.force_to_top = {"Sys"}  # Note capital S
    config.case_sensitive = True
    
    # Should not match because case doesn't match
    assert section_key("import sys", config) == "Bimport sys"
    
    config.case_sensitive = False
    # Should match because we ignore case
    assert section_key("import sys", config) == "Aimport sys"
    
    # Test 13: Multiple imports on one line
    config.force_to_top = set()
    config.case_sensitive = True
    config.order_by_type = False
    
    result = section_key("import sys, os, math", config)
    assert result == "Bimport sys, os, math"


# LLM-generated content at query #20
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("os", config) == "Bos"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"
    config.case_sensitive = True

    # Test force_to_top
    config.force_to_top = {"os"}
    assert module_key("os", config) == "Aos"
    config.force_to_top = set()

    # Test relative imports
    config.reverse_relative = False
    assert module_key("..module", config) == "B.._module"
    
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    
    # Test constants
    config.constants = {"MAX_VALUE"}
    assert module_key("MAX_VALUE", config, sub_imports=True) == "BAmax_value"
    
    # Test classes
    config.classes = {"MyClass"}
    assert module_key("MyClass", config, sub_imports=True) == "Bbmyclass"
    
    # Test variables
    config.variables = {"my_var"}
    assert module_key("my_var", config, sub_imports=True) == "Bcmy_var"
    
    # Test uppercase detection (issue #376)
    assert module_key("CONST", config, sub_imports=True) == "BAconst"
    
    # Test class detection by first letter
    assert module_key("ClassName", config, sub_imports=True) == "Bbclassname"
    
    # Test default prefix
    assert module_key("function_name", config, sub_imports=True) == "Bcfunction_name"

    config.order_by_type = False

    # Test length_sort
    config.length_sort = True
    assert module_key("os", config) == "B2:os"
    assert module_key("long_module_name", config) == "B16:long_module_name"
    config.length_sort = False

    # Test length_sort_straight with straight_import
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2:os"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"future"}
    assert module_key("os", config, section_name="future") == "B2:os"

    # Test ignore_case parameter
    assert module_key("OS", config, ignore_case=True) == "Bos"
    assert module_key("Os", config, ignore_case=True) == "Bos"

    # Test combined scenarios
    config.force_to_top = {"django"}
    config.length_sort = True
    config.order_by_type = True
    config.constants = {"SETTINGS"}
    
    result = module_key("django", config, sub_imports=True)
    assert result.startswith("A")  # Force to top
    assert ":" in result  # Length sort
    
    result = module_key("SETTINGS", config, sub_imports=True)
    assert result.startswith("BA")  # Force to top + constant prefix


# LLM-generated content at query #21
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(
            self,
            reverse_relative=False,
            order_by_type=False,
            constants=None,
            classes=None,
            variables=None,
            case_sensitive=True,
            length_sort=False,
            length_sort_straight=False,
            length_sort_sections=None,
            force_to_top=None,
        ):
            self.reverse_relative = reverse_relative
            self.order_by_type = order_by_type
            self.constants = constants or set()
            self.classes = classes or set()
            self.variables = variables or set()
            self.case_sensitive = case_sensitive
            self.length_sort = length_sort
            self.length_sort_straight = length_sort_straight
            self.length_sort_sections = length_sort_sections or set()
            self.force_to_top = force_to_top or set()

    # Test basic module name
    config = MockConfig()
    assert module_key("module", config) == "Bmodule"

    # Test relative imports with reverse_relative=False
    config = MockConfig(reverse_relative=False)
    assert module_key("..module", config) == "B_.. module"

    # Test relative imports with reverse_relative=True
    config = MockConfig(reverse_relative=True)
    assert module_key("..module", config) == "B .. module"

    # Test ignore_case parameter
    config = MockConfig()
    assert module_key("MODULE", config, ignore_case=True) == "Bmodule"
    assert module_key("Module", config, ignore_case=True) == "Bmodule"

    # Test case_sensitive=False in config
    config = MockConfig(case_sensitive=False)
    assert module_key("MODULE", config) == "Bmodule"
    assert module_key("Module", config) == "Bmodule"

    # Test order_by_type with sub_imports
    config = MockConfig(
        order_by_type=True,
        constants={"CONSTANT"},
        classes={"Class"},
        variables={"variable"},
    )
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    assert module_key("Other", config, sub_imports=True) == "BCOther"

    # Test order_by_type with uppercase module (issue #376)
    config = MockConfig(order_by_type=True)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    assert module_key("A", config, sub_imports=True) == "BCA"  # Single char not treated as constant

    # Test order_by_type with class detection
    config = MockConfig(order_by_type=True)
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    assert module_key("lowercase", config, sub_imports=True) == "BClowercase"

    # Test length_sort
    config = MockConfig(length_sort=True)
    assert module_key("module", config) == "B6:module"
    assert module_key("longmodule", config) == "B10:longmodule"

    # Test length_sort_straight with straight_import=True
    config = MockConfig(length_sort_straight=True)
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"

    # Test length_sort_sections
    config = MockConfig(length_sort_sections={"test_section"})
    assert module_key("module", config, section_name="test_section") == "B6:module"
    assert module_key("module", config, section_name="other_section") == "Bmodule"

    # Test force_to_top
    config = MockConfig(force_to_top={"top_module"})
    assert module_key("top_module", config) == "Atop_module"
    assert module_key("other_module", config) == "Bother_module"

    # Test combined features
    config = MockConfig(
        order_by_type=True,
        constants={"MY_CONST"},
        length_sort=True,
        force_to_top={"MY_CONST"},
    )
    result = module_key("MY_CONST", config, sub_imports=True)
    assert result == "AA8:MY_CONST"

    # Test with numeric module names
    config = MockConfig()
    assert module_key("module123", config) == "Bmodule123"

    # Test empty module name
    config = MockConfig()
    assert module_key("", config) == "B"


# LLM-generated content at query #22
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name handling
    config = Mock(
        reverse_relative=False,
        order_by_type=False,
        constants=set(),
        classes=set(),
        variables=set(),
        case_sensitive=True,
        length_sort=False,
        length_sort_straight=False,
        length_sort_sections=set(),
        force_to_top=set(),
    )
    
    assert module_key("module", config) == "Bmodule"
    
    # Test relative imports with reverse_relative=False
    assert module_key("..module", config) == "B..module"
    
    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"
    
    # Test case sensitivity
    config.case_sensitive = True
    assert module_key("Module", config) == "BModule"
    assert module_key("module", config) == "Bmodule"
    
    config.case_sensitive = False
    assert module_key("Module", config) == "Bmodule"
    assert module_key("module", config) == "Bmodule"
    
    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}
    
    # Test constants
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    
    # Test classes
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    
    # Test variables
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    
    # Test uppercase detection for constants
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    
    # Test class detection by first letter
    assert module_key("ClassName", config, sub_imports=True) == "BBClassName"
    
    # Test default prefix for other cases
    assert module_key("function", config, sub_imports=True) == "BCfunction"
    
    # Test length_sort
    config.length_sort = True
    assert module_key("module", config) == "B3:module"
    assert module_key("longmodule", config) == "B10:longmodule"
    
    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"test"}
    assert module_key("module", config, section_name="test") == "B6:module"
    assert module_key("module", config, section_name="other") == "Bmodule"
    
    # Test force_to_top
    config.force_to_top = {"special"}
    assert module_key("special", config) == "Aspecial"
    assert module_key("regular", config) == "Bregular"
    
    # Test ignore_case parameter
    config.case_sensitive = True
    assert module_key("Module", config, ignore_case=True) == "Bmodule"
    assert module_key("module", config, ignore_case=True) == "Bmodule"
    
    # Test combined scenarios
    config.order_by_type = True
    config.length_sort = True
    config.force_to_top = {"CONSTANT"}
    
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result.startswith("AA")  # Force to top + constant prefix
    assert "CONSTANT" in result
    
    # Test with empty module name
    assert module_key("", config) == "B0:"
    
    # Test with numeric module names
    assert module_key("123", config) == "B3:123"
    
    # Test with special characters
    assert module_key("module-name", config) == "B11:module-name"


# LLM-generated content at query #23
#--------------------------

```python
def test_section_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.group_by_package = False
            self.lexicographical = False
            self.sort_relative_in_force_sorted_sections = False
            self.force_to_top = set()
            self.honor_case_in_force_sorted_sections = False
            self.case_sensitive = True
            self.order_by_type = True
            self.length_sort = False

    config = MockConfig()

    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"

    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"

    # Test case sensitivity
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from SYS import PATH", config) == "Bfrom sys import path"

    # Test lexicographical mode
    config.lexicographical = True
    config.case_sensitive = True
    config.order_by_type = True
    assert section_key("import os.path", config) == "Bos.path"
    assert section_key("from os import path", config) == "Bos.path"

    # Test group_by_package mode
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from os import path", config) == "Bfrom os"

    # Test length_sort
    config.length_sort = True
    config.group_by_package = False
    assert section_key("import a", config) == "B11import a"
    assert section_key("import abc", config) == "B15import abc"

    # Test reverse_relative with sort_relative_in_force_sorted_sections
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    config.length_sort = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"

    # Test honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    line = "from MODULE import ClassName"
    result = section_key(line, config)
    assert "MODULE" in result
    assert "classname" in result

    # Test mixed case handling
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import MixedCase", config) == "Bimport mixedcase"

    # Test relative imports without special handling
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from .module import func", config) == "Bfrom .module import func"
    assert section_key("from ..package import Class", config) == "Bfrom ..package import Class"

    # Test empty or whitespace lines
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "

    # Test with multiple imports on one line
    assert section_key("import os, sys, math", config) == "Bimport os, sys, math"


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from os import path", config) == "Bfrom os import path"
    
    # Test 2: Force to top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("from os import path", config) == "Afrom os import path"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test 3: Case sensitivity and order_by_type interactions
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("IMPORT OS", config) == "Bimport os"
    assert section_key("From OS Import Path", config) == "Bfrom os import path"
    
    # Test 4: honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("from OS import PATH", config) == "Bfrom OS import path"
    
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("from OS import PATH", config) == "Bfrom os import PATH"
    
    # Test 5: Relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 6: sort_relative_in_force_sorted_sections with reverse_relative
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 7: group_by_package functionality
    config.group_by_package = True
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    assert section_key("from package import module", config) == "Bfrom package"
    assert section_key("from package.subpackage import module", config) == "Bfrom package.subpackage"
    
    # Test 8: lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("import os.path", config) == "Bimport os.path"
    assert section_key("from os import path", config) == "Bos.path"
    
    # Test 9: length_sort functionality
    config.lexicographical = False
    config.length_sort = True
    assert section_key("import a", config) == "B11import a"
    assert section_key("import abc", config) == "B13import abc"
    
    # Test 10: Complex scenario with multiple configurations
    config = Mock(
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=True,
        group_by_package=True,
        lexicographical=True,
        force_to_top={"django"},
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True,
        length_sort=False
    )
    
    assert section_key("from django import settings", config) == "Afrom django"
    assert section_key("from . import models", config) == "Bfrom . import models"
    
    # Test 11: Edge case - empty line
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    assert section_key("", config) == "B"
    
    # Test 12: Multiple imports in force_to_top
    config.force_to_top = {"os", "sys"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Aimport sys"
    assert section_key("import django", config) == "Bimport django"


# LLM-generated content at query #2
#--------------------------

```python
def test_naturally():
    # Test basic alphabetical sorting
    assert naturally(["b", "a", "c"]) == ["a", "b", "c"]
    
    # Test reverse sorting
    assert naturally(["b", "a", "c"], reverse=True) == ["c", "b", "a"]
    
    # Test natural sorting with numbers
    assert naturally(["item2", "item10", "item1"]) == ["item1", "item2", "item10"]
    
    # Test natural sorting with mixed alphanumeric
    assert naturally(["a10", "a2", "a1"]) == ["a1", "a2", "a10"]
    
    # Test with custom key function
    result = naturally(["bbb", "aa", "c"], key=len)
    assert result == ["c", "aa", "bbb"] or result == ["bbb", "aa", "c"]
    
    # Test with empty list
    assert naturally([]) == []
    
    # Test with single element
    assert naturally(["single"]) == ["single"]
    
    # Test with duplicate elements
    assert naturally(["b", "a", "b", "c"]) == ["a", "b", "b", "c"]
    
    # Test complex natural sorting
    items = ["file10.txt", "file2.txt", "file1.txt", "file20.txt"]
    assert naturally(items) == ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]
    
    # Test with numbers at beginning
    assert naturally(["10a", "2a", "1a"]) == ["1a", "2a", "10a"]
    
    # Test with only numbers
    assert naturally(["10", "2", "1", "20"]) == ["1", "2", "10", "20"]
    
    # Test natural sorting with reverse
    items = ["item2", "item10", "item1"]
    assert naturally(items, reverse=True) == ["item10", "item2", "item1"]
    
    # Test that key function is applied before natural sorting
    def custom_key(x):
        return x[1:]  # Remove first character
    
    result = naturally(["a10", "b2", "c1"], key=custom_key)
    # After removing first char: "10", "2", "1" -> natural sort gives "1", "2", "10"
    # So original items should be: "c1", "b2", "a10"
    assert result == ["c1", "b2", "a10"]


# LLM-generated content at query #3
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test reverse_relative with sort_relative_in_force_sorted_sections=False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    line = "from . import module"
    result = section_key(line, config)
    assert result == "Bfrom . import module"
    
    # Test group_by_package
    config.group_by_package = True
    config.reverse_relative = False
    line = "from package.subpackage import module"
    result = section_key(line, config)
    assert result == "Bfrom package.subpackage"
    
    # Test lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    line = "from package import module"
    result = section_key(line, config)
    assert result == "Bpackage.module"
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    line = "from ... import module"
    result = section_key(line, config)
    assert result == "Bfrom ... import module"
    
    # Test case sensitivity with honor_case_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    line = "from MyPackage import MyClass"
    result = section_key(line, config)
    assert result == "Bfrom mypackage import MyClass"
    
    # Test without order_by_type (should lowercase everything)
    config.order_by_type = False
    line = "from MyPackage import MyClass"
    result = section_key(line, config)
    assert result == "Bfrom mypackage import myclass"
    
    # Test length_sort
    config.length_sort = True
    config.order_by_type = True
    config.case_sensitive = True
    line = "import a"
    result = section_key(line, config)
    assert result == "B8import a"
    
    # Test with relative imports and various configurations
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    line = "from .. import module"
    result = section_key(line, config)
    assert result == "Bfrom .. import module"
    
    # Test multiple dots in relative import
    line = "from ..... import deep_module"
    result = section_key(line, config)
    assert result == "Bfrom ..... import deep_module"
    
    # Test mixed case with different configs
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    line = "import MixedCaseModule"
    result = section_key(line, config)
    assert result == "Bimport mixedcasemodule"
    
    # Test import statement (not from)
    line = "import package.module"
    result = section_key(line, config)
    assert result == "Bimport package.module"
    
    # Test empty line (edge case)
    line = ""
    result = section_key(line, config)
    assert result == "B"


# LLM-generated content at query #4
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()
    
    result = module_key("module", config)
    assert result == "Bmodule"
    
    # Test with relative imports
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.. module"
    
    config.reverse_relative = False
    result = module_key("...module", config)
    assert result == "B..._module"
    
    # Test ignore_case parameter
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"
    
    # Test case_sensitive config
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"
    
    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.case_sensitive = True
    
    # Test constants
    config.constants = {"CONSTANT"}
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"
    
    # Test classes
    config.classes = {"ClassName"}
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"
    
    # Test variables
    config.variables = {"variable"}
    result = module_key("variable", config, sub_imports=True)
    assert result == "BCvariable"
    
    # Test uppercase detection (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"
    
    # Test class detection by first letter
    result = module_key("AnotherClass", config, sub_imports=True)
    assert result == "BBAnotherClass"
    
    # Test default prefix for other names
    result = module_key("function", config, sub_imports=True)
    assert result == "BCfunction"
    
    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"
    
    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"
    
    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"test_section"}
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"
    
    # Test force_to_top
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"
    
    # Test combined scenarios
    config.order_by_type = True
    config.length_sort = True
    config.force_to_top = {"ClassName"}
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "AB9:ClassName"


# LLM-generated content at query #5
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic import line with default config
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    config.length_sort = False
    
    line = "import os"
    result = section_key(line, config)
    assert result == "Bos"
    
    # Test 2: From import line
    line = "from collections import defaultdict"
    result = section_key(line, config)
    assert result == "Bcollections import defaultdict"
    
    # Test 3: Force to top module
    config.force_to_top = ["os"]
    line = "import os"
    result = section_key(line, config)
    assert result == "Aos"
    
    # Test 4: Case insensitive with order_by_type=False
    config.order_by_type = False
    config.case_sensitive = False
    config.force_to_top = []
    line = "import OS"
    result = section_key(line, config)
    assert result == "Bos"
    
    # Test 5: Lexicographical sorting
    config.lexicographical = True
    config.case_sensitive = True
    line = "from collections import defaultdict"
    result = section_key(line, config)
    assert result == "Bcollections.defaultdict"
    
    # Test 6: Group by package
    config.lexicographical = False
    config.group_by_package = True
    line = "from collections import defaultdict, OrderedDict"
    result = section_key(line, config)
    assert result == "Bcollections"
    
    # Test 7: Length sort
    config.group_by_package = False
    config.length_sort = True
    line = "import a"
    result = section_key(line, config)
    assert result == "B1a"
    
    line = "import abc"
    result = section_key(line, config)
    assert result == "B3abc"
    
    # Test 8: Relative imports with reverse_relative
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    line = "from . import module"
    result = section_key(line, config)
    assert result == "B. import module"
    
    # Test 9: Relative imports without reverse_relative
    config.reverse_relative = False
    line = "from .. import module"
    result = section_key(line, config)
    assert result == "B.. import module"
    
    # Test 10: sort_relative_in_force_sorted_sections with reverse_relative
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    line = "from ... import module"
    result = section_key(line, config)
    assert result == "B... import module"
    
    # Test 11: sort_relative_in_force_sorted_sections without reverse_relative
    config.reverse_relative = False
    line = "from .. import module"
    result = section_key(line, config)
    assert result == "B.._import module"
    
    # Test 12: honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    line = "from COLLECTIONS import defaultdict"
    result = section_key(line, config)
    assert result == "BCOLLECTIONS import defaultdict"
    
    # Test 13: honor_case_in_force_sorted_sections with case_sensitive=False
    config.case_sensitive = False
    line = "from COLLECTIONS import defaultdict"
    result = section_key(line, config)
    assert result == "Bcollections import defaultdict"
    
    # Test 14: honor_case_in_force_sorted_sections with order_by_type=False
    config.case_sensitive = True
    config.order_by_type = False
    line = "from collections import DEFAULTDICT"
    result = section_key(line, config)
    assert result == "Bcollections import defaultdict"
    
    # Test 15: Complex import with multiple conditions
    config.force_to_top = ["sys"]
    config.length_sort = True
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = False
    line = "import sys"
    result = section_key(line, config)
    assert result == "A3sys"
    
    # Test 16: Empty line
    line = ""
    result = section_key(line, config)
    assert result == "B0"


# LLM-generated content at query #6
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic functionality with default config
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False

    # Test regular import
    assert section_key("import os", config) == "Bimport os"
    
    # Test from import
    assert section_key("from os import path", config) == "Bfrom os import path"
    
    # Test force_to_top
    config.force_to_top = ["os"]
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test case sensitivity
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Aimport os"
    assert section_key("import Sys", config) == "Bimport sys"
    
    # Test order_by_type = False
    config.order_by_type = False
    config.case_sensitive = True
    assert section_key("import OS", config) == "Aimport os"
    assert section_key("import Sys", config) == "Bimport sys"
    
    # Test length_sort
    config.length_sort = True
    config.order_by_type = True
    config.case_sensitive = True
    config.force_to_top = []
    assert section_key("import a", config) == "B9import a"
    assert section_key("import abc", config) == "B11import abc"
    
    # Test lexicographical
    config.length_sort = False
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bfrom package"
    
    # Test relative imports with reverse_relative
    config.group_by_package = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.force_to_top = []
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from OS import PATH", config) == "Bfrom OS import path"
    
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from OS import PATH", config) == "Bfrom os import PATH"
    
    # Test mixed case scenarios
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    assert section_key("import os", config) == "Bimport os"
    assert section_key("import OS", config) == "Bimport OS"
    
    # Test with force_to_top and other config combinations
    config.force_to_top = ["django"]
    config.length_sort = True
    config.case_sensitive = False
    assert section_key("import django", config) == "A13import django"
    assert section_key("import Django", config) == "A13import django"


# LLM-generated content at query #7
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic section assignment
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top={"sys", "os"},
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Regular import should get section B
    assert section_key("import numpy", config) == "Bimport numpy"
    
    # Force to top import should get section A
    assert section_key("import sys", config) == "Aimport sys"
    assert section_key("import os", config) == "Aimport os"
    
    # Test 2: Case sensitivity with order_by_type
    config.case_sensitive = False
    config.order_by_type = True
    config.honor_case_in_force_sorted_sections = True
    
    # Should lowercase the whole line when order_by_type is True
    assert section_key("import MyModule", config) == "Bimport mymodule"
    
    # Test 3: Case sensitivity without order_by_type
    config.case_sensitive = True
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = True
    
    # Should lowercase names part only
    result = section_key("from package import MyClass, my_function", config)
    assert result == "Bfrom package import myclass, my_function"
    
    # Test 4: lexicographical mode
    config.lexicographical = True
    config.order_by_type = True
    config.case_sensitive = True
    
    # Should transform import lines
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b.c", config) == "Ba.b.c"
    
    # Test 5: group_by_package mode
    config.lexicographical = False
    config.group_by_package = True
    
    # Should keep only the from part
    assert section_key("from package import something", config) == "Bfrom package"
    
    # Test 6: reverse_relative with sort_relative_in_force_sorted_sections=False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    
    # Should convert "from ." to "from . "
    assert section_key("from . import mod", config) == "Bfrom . import mod"
    
    # Test 7: sort_relative_in_force_sorted_sections=True
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    
    # Should add space after dots
    assert section_key("from .. import mod", config) == "Bfrom .. import mod"
    
    # Test 8: length_sort enabled
    config.length_sort = True
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    
    # Should include length in key
    result = section_key("import abc", config)
    assert result.startswith("B10")  # "import abc" is 10 chars
    
    # Test 9: Mixed case handling with from/import
    config.length_sort = False
    config.case_sensitive = False
    config.order_by_type = False
    config.honor_case_in_force_sorted_sections = False
    
    # Should lowercase everything
    assert section_key("FROM Package IMPORT Class", config) == "Bfrom package import class"
    
    # Test 10: Force to top with complex line
    config.case_sensitive = True
    config.order_by_type = True
    config.force_to_top = {"django"}
    
    # Should get section A for force_to_top
    assert section_key("from django.conf import settings", config) == "Afrom django.conf import settings"
    
    # Test 11: Empty or whitespace lines
    config.force_to_top = set()
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "
    
    # Test 12: Import statement without from
    assert section_key("import module.submodule", config) == "Bmodule.submodule"
    
    # Test 13: Multiple imports in one line
    result = section_key("from mod import a, b, c", config)
    assert result == "Bfrom mod import a, b, c"


# LLM-generated content at query #8
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic functionality with default config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()

    result = module_key("module_name", config)
    assert result == "Bmodule_name"

    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    result = module_key("..module", config)
    assert result == "B.. module"

    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"

    # Test ignore_case parameter
    config.reverse_relative = False
    result = module_key("ModuleName", config, ignore_case=True)
    assert result == "Bmodulename"

    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("ModuleName", config)
    assert result == "Bmodulename"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.case_sensitive = True
    config.constants = {"CONSTANT"}
    config.classes = {"ClassName"}
    config.variables = {"variable_name"}

    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"

    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"

    result = module_key("variable_name", config, sub_imports=True)
    assert result == "BCvariable_name"

    # Test uppercase module name (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"

    # Test class detection by first uppercase letter
    result = module_key("SomeClass", config, sub_imports=True)
    assert result == "BBSomeClass"

    # Test default prefix for regular modules
    result = module_key("regular_module", config, sub_imports=True)
    assert result == "BCregular_module"

    # Test length_sort
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

    # Test length_sort_straight with straight_import
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

    # Test length_sort_sections
    config.length_sort = False
    config.length_sort_straight = False
    result = module_key("module", config, section_name="test_section")
    assert result == "Bmodule"

    config.length_sort_sections = {"test_section"}
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"

    # Test force_to_top
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

    # Test combined scenarios
    config.order_by_type = True
    config.length_sort = True
    config.force_to_top = {"ClassName"}
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "ABB8:ClassName"

    # Test with empty module name
    result = module_key("", config)
    assert result.startswith("B")

    # Test numeric module names
    result = module_key("123", config, sub_imports=True)
    assert result == "BC123"

    # Test module name with special characters
    result = module_key("module-name", config)
    assert result == "Bmodule-name"


# LLM-generated content at query #9
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("os", config) == "Bos"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"
    config.case_sensitive = True

    # Test force_to_top
    config.force_to_top = {"os"}
    assert module_key("os", config) == "Aos"
    config.force_to_top = set()

    # Test relative imports
    config.reverse_relative = False
    assert module_key(".os", config) == "B_os"
    
    config.reverse_relative = True
    assert module_key(".os", config) == "B .os"
    config.reverse_relative = False

    # Test length_sort
    config.length_sort = True
    assert module_key("os", config) == "B2:os"
    config.length_sort = False

    # Test length_sort_straight
    config.length_sort_straight = True
    assert module_key("os", config, straight_import=True) == "B2:os"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"test"}
    assert module_key("os", config, section_name="test") == "B2:os"
    config.length_sort_sections = set()

    # Test order_by_type with sub_imports
    config.order_by_type = True
    
    # Test constants
    config.constants = {"MAX_SIZE"}
    assert module_key("MAX_SIZE", config, sub_imports=True) == "BAmax_size"
    
    # Test classes
    config.classes = {"MyClass"}
    assert module_key("MyClass", config, sub_imports=True) == "Bbmyclass"
    
    # Test variables
    config.variables = {"my_var"}
    assert module_key("my_var", config, sub_imports=True) == "Bcmy_var"
    
    # Test uppercase module (issue #376)
    assert module_key("CONST", config, sub_imports=True) == "Baconst"
    
    # Test class by naming convention
    assert module_key("ClassName", config, sub_imports=True) == "Bbclassname"
    
    # Test default (other)
    assert module_key("function_name", config, sub_imports=True) == "Bcfunction_name"
    
    config.order_by_type = False

    # Test ignore_case parameter
    assert module_key("OS", config, ignore_case=True) == "Bos"

    # Test combined scenarios
    config.force_to_top = {"django"}
    config.length_sort = True
    config.order_by_type = True
    config.constants = {"SETTINGS"}
    
    result = module_key("django", config, sub_imports=True)
    assert result.startswith("A")
    
    result = module_key("SETTINGS", config, sub_imports=True)
    assert "A" in result  # Should have 'A' prefix for constants
    
    # Test with numeric module names
    assert module_key("module123", config) == "Bmodule123"


# LLM-generated content at query #10
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic section assignment
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False
    )
    
    # Regular import should get section B
    assert section_key("import os", config) == "Bos"
    assert section_key("from os import path", config) == "Bos.path"
    
    # Test 2: Force to top
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aos"
    assert section_key("from os import path", config) == "Aos.path"
    
    # Test 3: Case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    
    # Test 4: Length sort
    config.length_sort = True
    config.case_sensitive = True
    assert section_key("import os", config) == "B2os"
    assert section_key("import pathlib", config) == "B7pathlib"
    
    # Test 5: Lexicographical mode
    config.length_sort = False
    config.lexicographical = True
    assert section_key("import os.path", config) == "Bos.path"
    assert section_key("from os import path", config) == "Bos.path"
    
    # Test 6: Group by package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from os import path, sys", config) == "Bos"
    assert section_key("from collections import defaultdict", config) == "Bcollections"
    
    # Test 7: Relative imports with reverse_relative
    config.group_by_package = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "B. import module"
    assert section_key("from ..parent import child", config) == "B.. parent import child"
    
    # Test 8: sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "B._import module"
    assert section_key("from ..parent import child", config) == "B.._parent import child"
    
    # Test 9: honor_case_in_force_sorted_sections with case_sensitive != order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    
    # Case sensitive module, lowercase names
    result = section_key("from OS import PATH, SYSTEM", config)
    assert result == "BOS import path, system"
    
    # Test 10: honor_case_in_force_sorted_sections with case_sensitive=False, order_by_type=True
    config.case_sensitive = False
    config.order_by_type = True
    result = section_key("from OS import PATH", config)
    assert result == "Bos import PATH"
    
    # Test 11: Simple import line (no "from")
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bos"
    
    # Test 12: Mixed case handling
    config.case_sensitive = True
    assert section_key("import OsPath", config) == "BOsPath"
    assert section_key("import osPath", config) == "BosPath"
    
    # Test 13: Empty or whitespace lines
    config.case_sensitive = False
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B"
    
    # Test 14: Complex import with multiple names
    config.lexicographical = True
    result = section_key("from package.subpackage import Class, function, CONSTANT", config)
    # In lexicographical mode, "from package.subpackage import" becomes "package.subpackage."
    assert "package.subpackage." in result
    
    # Test 15: Force to top with length sort
    config.lexicographical = False
    config.length_sort = True
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "A2os"
    assert section_key("import pathlib", config) == "B7pathlib"


# LLM-generated content at query #11
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic config with default values
    config = Mock()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = []
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True
    config.length_sort = False

    # Test basic import line
    assert section_key("import os", config) == "Bos"
    assert section_key("import sys", config) == "Bsys"
    assert section_key("from os import path", config) == "Bos"

    # Test force_to_top functionality
    config.force_to_top = ["os"]
    assert section_key("import os", config) == "Aos"
    assert section_key("import sys", config) == "Bsys"
    config.force_to_top = []

    # Test case sensitivity
    config.case_sensitive = False
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    config.case_sensitive = True

    # Test order_by_type
    config.order_by_type = False
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    config.order_by_type = True

    # Test length_sort
    config.length_sort = True
    assert section_key("import a", config) == "B1a"
    assert section_key("import abc", config) == "B3abc"
    config.length_sort = False

    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "B. import module"
    config.reverse_relative = False

    # Test group_by_package
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bpackage"
    config.group_by_package = False

    # Test lexicographical
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    config.lexicographical = False

    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = False
    assert section_key("from .. import module", config) == "B.._import module"
    config.reverse_relative = True
    assert section_key("from .. import module", config) == "B.. import module"
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False

    # Test honor_case_in_force_sorted_sections with mixed settings
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("from OS import PATH", config) == "Bos import PATH"
    assert section_key("import OS", config) == "Bos"

    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("from OS import PATH", config) == "BOS import path"
    assert section_key("import OS", config) == "BOS"

    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = True

    # Test complex import statements
    assert section_key("from package.subpackage import ClassName", config) == "Bpackage.subpackage"
    assert section_key("import package.subpackage.module", config) == "Bpackage.subpackage.module"

    # Test with actual force_to_top elements
    config.force_to_top = ["sys", "os"]
    assert section_key("import sys", config) == "Asys"
    assert section_key("import os", config) == "Aos"
    assert section_key("import json", config) == "Bjson"


# LLM-generated content at query #12
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(
            self,
            reverse_relative=False,
            order_by_type=False,
            constants=None,
            classes=None,
            variables=None,
            case_sensitive=True,
            length_sort=False,
            length_sort_straight=False,
            length_sort_sections=None,
            force_to_top=None,
        ):
            self.reverse_relative = reverse_relative
            self.order_by_type = order_by_type
            self.constants = constants or set()
            self.classes = classes or set()
            self.variables = variables or set()
            self.case_sensitive = case_sensitive
            self.length_sort = length_sort
            self.length_sort_straight = length_sort_straight
            self.length_sort_sections = length_sort_sections or set()
            self.force_to_top = force_to_top or set()

    # Test basic module name
    config = MockConfig()
    assert module_key("os", config) == "Bos"

    # Test force_to_top
    config = MockConfig(force_to_top={"os"})
    assert module_key("os", config) == "Aos"

    # Test relative imports with reverse_relative=False
    config = MockConfig(reverse_relative=False)
    assert module_key("..module", config) == "B.._module"

    # Test relative imports with reverse_relative=True
    config = MockConfig(reverse_relative=True)
    assert module_key("..module", config) == "B.. module"

    # Test ignore_case parameter
    config = MockConfig()
    assert module_key("OS", config, ignore_case=True) == "Bos"
    assert module_key("Os", config, ignore_case=True) == "Bos"

    # Test case_sensitive=False
    config = MockConfig(case_sensitive=False)
    assert module_key("OS", config) == "Bos"
    assert module_key("Os", config) == "Bos"

    # Test order_by_type with sub_imports
    config = MockConfig(order_by_type=True)
    # Test constants
    config.constants = {"MAX_SIZE"}
    assert module_key("MAX_SIZE", config, sub_imports=True) == "BAmax_size"
    # Test classes
    config.classes = {"MyClass"}
    assert module_key("MyClass", config, sub_imports=True) == "BBmyclass"
    # Test variables
    config.variables = {"my_var"}
    assert module_key("my_var", config, sub_imports=True) == "BCmy_var"
    # Test uppercase module (issue #376)
    assert module_key("SOME_CONST", config, sub_imports=True) == "BAsome_const"
    # Test class-like module (first letter uppercase)
    assert module_key("SomeModule", config, sub_imports=True) == "BBsomemodule"
    # Test default (prefix C)
    assert module_key("regular_module", config, sub_imports=True) == "BCregular_module"

    # Test length_sort
    config = MockConfig(length_sort=True)
    assert module_key("os", config) == "B2:os"
    assert module_key("json", config) == "B4:json"

    # Test length_sort_straight with straight_import=True
    config = MockConfig(length_sort_straight=True)
    assert module_key("os", config, straight_import=True) == "B2:os"
    assert module_key("json", config, straight_import=True) == "B4:json"

    # Test length_sort_sections
    config = MockConfig(length_sort_sections={"future"})
    assert module_key("os", config, section_name="future") == "B2:os"
    assert module_key("json", config, section_name="future") == "B4:json"

    # Test combination of factors
    config = MockConfig(
        order_by_type=True,
        constants={"VERSION"},
        force_to_top={"sys"},
        length_sort=True,
    )
    # Force to top with length sort
    assert module_key("sys", config, sub_imports=True) == "A6:sys"
    # Constant with length sort
    assert module_key("VERSION", config, sub_imports=True) == "BA7:version"
    # Regular module with length sort
    assert module_key("my_module", config, sub_imports=True) == "BC9:my_module"

    # Test with empty module name
    config = MockConfig()
    assert module_key("", config) == "B"

    # Test with numeric module name
    config = MockConfig()
    assert module_key("123", config) == "B123"

    # Test section_name as non-string
    config = MockConfig(length_sort_sections={"test"})
    assert module_key("module", config, section_name=123) == "Bmodule"


# LLM-generated content at query #13
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock

    # Test basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("import os", config) == "Bimport os"
    
    # Test order_by_type=False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from SYS import PATH", config) == "Bfrom sys import path"
    
    # Test lexicographical sorting
    config.lexicographical = True
    config.order_by_type = True
    assert section_key("import os.path", config) == "Bos.path"
    assert section_key("from os import path", config) == "Bos.path"
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from os import path", config) == "Bfrom os"
    assert section_key("from os.path import join", config) == "Bfrom os.path"
    
    # Test relative imports with reverse_relative
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    config.sort_relative_in_force_sorted_sections = False
    config.group_by_package = False
    config.reverse_relative = False
    
    line = "from MODULE import ClassName"
    result = section_key(line, config)
    assert result == "Bfrom MODULE import classname"
    
    # Test with case_sensitive=False and order_by_type=True
    config.case_sensitive = False
    config.order_by_type = True
    line = "from MODULE import ClassName"
    result = section_key(line, config)
    assert result == "Bfrom module import ClassName"
    
    # Test length_sort
    config.length_sort = True
    config.order_by_type = True
    config.case_sensitive = True
    line1 = "import a"
    line2 = "import abc"
    result1 = section_key(line1, config)
    result2 = section_key(line2, config)
    assert result1 == "B9import a"
    assert result2 == "B11import abc"
    
    # Test with force_to_top and length_sort
    config.force_to_top = {"abc"}
    result1 = section_key("import a", config)
    result2 = section_key("import abc", config)
    assert result1.startswith("B")
    assert result2.startswith("A")


# LLM-generated content at query #14
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without special handling
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = []
    config.force_to_top = set()
    
    result = module_key("module_name", config)
    assert result == "Bmodule_name"

    # Test force_to_top
    config.force_to_top = {"module_name"}
    result = module_key("module_name", config)
    assert result == "Amodule_name"

    # Test case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    result = module_key("Module_Name", config)
    assert result == "Bmodule_name"

    # Test length_sort
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

    # Test length_sort_straight
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = ["test_section"]
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"

    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    result = module_key("..module", config)
    assert result == "B.. module"

    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"

    # Test ignore_case parameter
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"ClassName"}
    config.variables = {"variable"}
    
    # Test constant
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"
    
    # Test class
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"
    
    # Test variable
    result = module_key("variable", config, sub_imports=True)
    assert result == "BCvariable"
    
    # Test uppercase module (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"
    
    # Test class by first letter uppercase
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"
    
    # Test default prefix for other modules
    result = module_key("other_module", config, sub_imports=True)
    assert result == "BCother_module"

    # Test combination of features
    config.force_to_top = {"special_module"}
    config.length_sort = True
    config.constants = {"MY_CONSTANT"}
    
    result = module_key("special_module", config, sub_imports=True)
    assert result == "A14:special_module"
    
    result = module_key("MY_CONSTANT", config, sub_imports=True)
    assert result == "BA11:MY_CONSTANT"


# LLM-generated content at query #15
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False
    )
    
    # Test basic import lines
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from os import path", config) == "Bfrom os import path"
    
    # Test 2: Force to top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("from os import path", config) == "Afrom os import path"
    assert section_key("import sys", config) == "Bimport sys"
    
    # Test 3: Case sensitivity variations
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("from OS import PATH", config) == "Bfrom os import path"
    
    # Test 4: Length sort
    config.length_sort = True
    config.case_sensitive = True
    config.order_by_type = True
    assert section_key("import a", config) == "B10import a"
    assert section_key("import abc", config) == "B11import abc"
    
    # Test 5: Relative imports with reverse_relative
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    
    # Test 6: Group by package
    config.reverse_relative = False
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bfrom package"
    
    # Test 7: Lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("from os import path", config) == "Bos.path"
    assert section_key("import os.path", config) == "Bos.path"
    
    # Test 8: sort_relative_in_force_sorted_sections with reverse_relative
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 9: honor_case_in_force_sorted_sections with different case_sensitive and order_by_type
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("from OS import PATH", config) == "Bfrom os import PATH"
    
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("from OS import PATH", config) == "Bfrom OS import path"
    
    # Test 10: Mixed scenarios
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = False
    config.force_to_top = {"sys"}
    config.length_sort = True
    assert section_key("import sys", config) == "A9import sys"
    assert section_key("import os", config) == "B9import os"


# LLM-generated content at query #16
#--------------------------

```python
def test_section_key():
    class MockConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Test basic sorting sections
    config = MockConfig(
        force_to_top=["django"],
        lexicographical=False,
        group_by_package=False,
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False,
    )
    
    # Test force_to_top section assignment
    assert section_key("import django", config) == "Aimport django"
    assert section_key("from django import something", config) == "Afrom django import something"
    assert section_key("import requests", config) == "Bimport requests"
    
    # Test lexicographical processing
    config.lexicographical = True
    assert section_key("import a.b.c", config) == "Bimport a.b.c"
    assert section_key("from a import b", config) == "Ba.b"
    
    # Test group_by_package
    config.lexicographical = False
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bfrom package"
    
    # Test length_sort
    config.group_by_package = False
    config.length_sort = True
    assert section_key("import a", config) == "B13import a"
    assert section_key("import abc", config) == "B15import abc"
    
    # Test case sensitivity with honor_case_in_force_sorted_sections
    config.length_sort = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("import MODULE", config) == "Bimport module"
    
    # Test order_by_type lowercase when False
    config.honor_case_in_force_sorted_sections = False
    config.order_by_type = False
    assert section_key("import MODULE", config) == "Bimport module"
    assert section_key("from PACKAGE import MODULE", config) == "Bfrom package import module"
    
    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config.order_by_type = True
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test relative imports without sort_relative_in_force_sorted_sections
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    
    # Test complex case with multiple conditions
    config = MockConfig(
        force_to_top=["zebra"],
        lexicographical=True,
        group_by_package=True,
        sort_relative_in_force_sorted_sections=True,
        reverse_relative=False,
        honor_case_in_force_sorted_sections=True,
        case_sensitive=False,
        order_by_type=True,
        length_sort=True,
    )
    
    result = section_key("from animal import zebra", config)
    assert result.startswith("B")
    
    # Test empty or whitespace lines
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        group_by_package=False,
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=True,
        length_sort=False,
    )
    
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "


# LLM-generated content at query #17
#--------------------------

```python
def test_section_key():
    from unittest.mock import Mock
    
    # Test 1: Basic sorting with default config
    config = Mock(
        sort_relative_in_force_sorted_sections=False,
        reverse_relative=False,
        group_by_package=False,
        lexicographical=False,
        force_to_top=set(),
        honor_case_in_force_sorted_sections=False,
        case_sensitive=True,
        order_by_type=False,
        length_sort=False
    )
    
    # Test basic import statements
    assert section_key("import os", config) == "Bimport os"
    assert section_key("from sys import path", config) == "Bfrom sys import path"
    
    # Test 2: Force to top functionality
    config.force_to_top = {"os", "sys"}
    assert section_key("import os", config) == "Aimport os"
    assert section_key("import sys", config) == "Aimport sys"
    assert section_key("import math", config) == "Bimport math"
    
    # Test 3: Case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport os"
    assert section_key("import os", config) == "Bimport os"
    
    # Test 4: Length sort
    config.length_sort = True
    config.case_sensitive = True
    assert section_key("import a", config) == "B9import a"
    assert section_key("import abc", config) == "B11import abc"
    
    # Test 5: Relative imports with reverse_relative
    config.length_sort = False
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = False
    assert section_key("from . import module", config) == "Bfrom . import module"
    assert section_key("from .. import module", config) == "Bfrom .. import module"
    
    # Test 6: Group by package
    config.group_by_package = True
    config.reverse_relative = False
    assert section_key("from package import module", config) == "Bfrom package"
    assert section_key("from package.sub import module", config) == "Bfrom package.sub"
    
    # Test 7: Lexicographical sorting
    config.group_by_package = False
    config.lexicographical = True
    assert section_key("from a import b", config) == "Ba.b"
    assert section_key("import a.b", config) == "Ba.b"
    
    # Test 8: Sort relative in force sorted sections
    config.lexicographical = False
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    assert section_key("from . import module", config) == "Bfrom . import module"
    config.reverse_relative = False
    assert section_key("from . import module", config) == "Bfrom ._import module"
    
    # Test 9: Honor case in force sorted sections
    config.sort_relative_in_force_sorted_sections = False
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("import OS", config) == "Bimport OS"
    assert section_key("from MODULE import CLASS", config) == "Bfrom MODULE import class"
    
    # Test 10: Complex case with multiple configurations
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = True
    config.force_to_top = {"django"}
    assert section_key("import django", config) == "Aimport django"
    assert section_key("import Django", config) == "Aimport django"
    
    # Test 11: Empty and whitespace lines
    config.force_to_top = set()
    config.case_sensitive = True
    config.order_by_type = False
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "
    
    # Test 12: Mixed case with force_to_top
    config.force_to_top = {"OS"}
    config.case_sensitive = True
    assert section_key("import OS", config) == "Aimport OS"
    assert section_key("import os", config) == "Bimport os"


# LLM-generated content at query #18
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without config
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()
    
    result = module_key("module_name", config)
    assert result == "Bmodule_name"

    # Test relative imports with reverse_relative=False
    config.reverse_relative = False
    result = module_key("..module", config)
    assert result == "B.. module"

    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"

    # Test ignore_case=True
    config.reverse_relative = False
    result = module_key("MODULE_NAME", config, ignore_case=True)
    assert result == "Bmodule_name"

    # Test ignore_case=False
    result = module_key("MODULE_NAME", config, ignore_case=False)
    assert result == "BMODULE_NAME"

    # Test sub_imports with order_by_type - constant
    config.order_by_type = True
    config.constants = {"module_name"}
    result = module_key("module_name", config, sub_imports=True)
    assert result == "BAmodule_name"

    # Test sub_imports with order_by_type - class
    config.constants = set()
    config.classes = {"ModuleName"}
    result = module_key("ModuleName", config, sub_imports=True)
    assert result == "BBModuleName"

    # Test sub_imports with order_by_type - variable
    config.classes = set()
    config.variables = {"module_name"}
    result = module_key("module_name", config, sub_imports=True)
    assert result == "BCmodule_name"

    # Test sub_imports with order_by_type - uppercase module (issue #376)
    config.variables = set()
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"

    # Test sub_imports with order_by_type - class by naming convention
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"

    # Test sub_imports with order_by_type - default (other)
    result = module_key("function_name", config, sub_imports=True)
    assert result == "BCfunction_name"

    # Test case_sensitive=False
    config.order_by_type = False
    config.case_sensitive = False
    result = module_key("ModuleName", config)
    assert result == "Bmodulename"

    # Test length_sort=True
    config.case_sensitive = True
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

    # Test length_sort_straight with straight_import=True
    config.length_sort = False
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

    # Test length_sort_sections
    config.length_sort_straight = False
    config.length_sort_sections = {"test_section"}
    result = module_key("module", config, section_name="test_section")
    assert result == "B6:module"

    # Test force_to_top
    config.length_sort = False
    config.length_sort_sections = set()
    config.force_to_top = {"module_name"}
    result = module_key("module_name", config)
    assert result == "Amodule_name"

    # Test combined: force_to_top with length_sort
    config.length_sort = True
    result = module_key("module_name", config)
    assert result == "A11:module_name"

    # Test combined: sub_imports with length_sort and order_by_type
    config.order_by_type = True
    config.constants = {"module_name"}
    config.force_to_top = set()
    result = module_key("module_name", config, sub_imports=True)
    assert result == "BA11:module_name"

    # Test with empty module name
    result = module_key("", config)
    assert result == "B0:"

    # Test with numeric module name
    result = module_key("123", config, sub_imports=True)
    assert result == "BC3:123"

    # Test with special characters
    result = module_key("module-name", config)
    assert result == "BC11:module-name"


# LLM-generated content at query #19
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name handling
    config = Mock()
    config.reverse_relative = False
    config.order_by_type = False
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.force_to_top = set()

    # Basic module name
    result = module_key("os", config)
    assert result == "Bos"

    # Test relative imports with reverse_relative=False
    result = module_key("..module", config)
    assert result == "B.. module"

    # Test relative imports with reverse_relative=True
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.._module"
    config.reverse_relative = False

    # Test ignore_case=True
    result = module_key("OS", config, ignore_case=True)
    assert result == "Bos"

    # Test case_sensitive=False
    config.case_sensitive = False
    result = module_key("OS", config)
    assert result == "Bos"
    config.case_sensitive = True

    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"ClassName"}
    config.variables = {"variable"}

    # Constant module
    result = module_key("CONSTANT", config, sub_imports=True)
    assert result == "BACONSTANT"

    # Class module
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"

    # Variable module
    result = module_key("variable", config, sub_imports=True)
    assert result == "BCvariable"

    # Uppercase module (issue #376)
    result = module_key("UPPERCASE", config, sub_imports=True)
    assert result == "BAUPPERCASE"

    # Module starting with uppercase
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

    # Regular module
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

    config.order_by_type = False

    # Test length_sort
    config.length_sort = True
    result = module_key("os", config)
    assert result == "B2:os"
    config.length_sort = False

    # Test length_sort_straight with straight_import=True
    config.length_sort_straight = True
    result = module_key("os", config, straight_import=True)
    assert result == "B2:os"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"test"}
    result = module_key("os", config, section_name="test")
    assert result == "B2:os"
    config.length_sort_sections = set()

    # Test force_to_top
    config.force_to_top = {"os"}
    result = module_key("os", config)
    assert result == "Aos"
    config.force_to_top = set()

    # Test combination of factors
    config.order_by_type = True
    config.constants = {"OS"}
    config.length_sort = True
    config.force_to_top = {"OS"}
    result = module_key("OS", config, sub_imports=True)
    assert result == "AA2:OS"


# LLM-generated content at query #20
#--------------------------

```python
def test_section_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.group_by_package = False
            self.lexicographical = False
            self.sort_relative_in_force_sorted_sections = False
            self.force_to_top = set()
            self.honor_case_in_force_sorted_sections = False
            self.case_sensitive = True
            self.order_by_type = True
            self.length_sort = False

    config = MockConfig()

    # Test basic import lines
    assert section_key("import os", config) == "Bos"
    assert section_key("from os import path", config) == "Bpath"
    assert section_key("import sys", config) == "Bsys"

    # Test force_to_top functionality
    config.force_to_top = {"os"}
    assert section_key("import os", config) == "Aos"
    assert section_key("from os import path", config) == "Apath"
    assert section_key("import sys", config) == "Bsys"

    # Test case sensitivity
    config.force_to_top = set()
    config.case_sensitive = False
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    config.case_sensitive = True

    # Test order_by_type
    config.order_by_type = False
    assert section_key("import OS", config) == "Bos"
    assert section_key("import os", config) == "Bos"
    config.order_by_type = True

    # Test length_sort
    config.length_sort = True
    assert section_key("import a", config) == "B1a"
    assert section_key("import abc", config) == "B3abc"
    config.length_sort = False

    # Test reverse_relative with sort_relative_in_force_sorted_sections
    config.reverse_relative = True
    config.sort_relative_in_force_sorted_sections = True
    assert section_key("from . import module", config) == "B. module"
    config.reverse_relative = False
    assert section_key("from .. import module", config) == "B.._module"
    config.sort_relative_in_force_sorted_sections = False

    # Test group_by_package
    config.group_by_package = True
    assert section_key("from package import module", config) == "Bpackage"
    config.group_by_package = False

    # Test lexicographical
    config.lexicographical = True
    assert section_key("from package import module", config) == "Bpackage.module"
    config.lexicographical = False

    # Test honor_case_in_force_sorted_sections
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = False
    config.order_by_type = True
    assert section_key("from PACKAGE import MODULE", config) == "Bpackage import MODULE"
    config.order_by_type = False
    assert section_key("from PACKAGE import MODULE", config) == "Bpackage import module"
    config.case_sensitive = True
    config.order_by_type = True
    assert section_key("from PACKAGE import MODULE", config) == "BPACKAGE import MODULE"
    config.honor_case_in_force_sorted_sections = False

    # Test mixed case handling
    config.case_sensitive = False
    assert section_key("import MixedCase", config) == "Bmixedcase"
    config.case_sensitive = True
    assert section_key("import MixedCase", config) == "BMixedCase"

    # Test with multiple imports on one line
    assert section_key("import os, sys, math", config) == "Bos, sys, math"

    # Test empty and whitespace lines
    assert section_key("", config) == "B"
    assert section_key("   ", config) == "B   "

    # Test edge cases
    config.force_to_top = {"sys"}
    config.length_sort = True
    assert section_key("import sys", config) == "A3sys"
    assert section_key("import os", config) == "B2os"


# LLM-generated content at query #21
#--------------------------

```python
def test_module_key():
    class MockConfig:
        def __init__(self):
            self.reverse_relative = False
            self.order_by_type = False
            self.constants = set()
            self.classes = set()
            self.variables = set()
            self.case_sensitive = True
            self.length_sort = False
            self.length_sort_straight = False
            self.length_sort_sections = set()
            self.force_to_top = set()

    config = MockConfig()

    # Test basic module name
    assert module_key("module", config) == "Bmodule"

    # Test case sensitivity
    config.case_sensitive = False
    assert module_key("Module", config) == "Bmodule"
    config.case_sensitive = True

    # Test relative imports
    config.reverse_relative = False
    assert module_key("..module", config) == "B_.._module"
    
    config.reverse_relative = True
    assert module_key("..module", config) == "B .. module"

    # Test force_to_top
    config.force_to_top = {"important"}
    assert module_key("important", config) == "Aimportant"
    assert module_key("other", config) == "Bother"

    # Test length_sort
    config.length_sort = True
    assert module_key("abc", config) == "B3:abc"
    assert module_key("abcd", config) == "B4:abcd"
    config.length_sort = False

    # Test length_sort_straight
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"special"}
    assert module_key("module", config, section_name="special") == "B6:module"
    assert module_key("module", config, section_name="other") == "Bmodule"

    # Test order_by_type with sub_imports
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}
    
    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    
    # Test uppercase detection (issue #376)
    assert module_key("UPPERCASE", config, sub_imports=True) == "BAUPPERCASE"
    
    # Test class detection by first letter
    assert module_key("MyClass", config, sub_imports=True) == "BBMyClass"
    
    # Test default prefix for others
    assert module_key("function", config, sub_imports=True) == "BCfunction"

    # Test ignore_case parameter
    assert module_key("Module", config, sub_imports=True, ignore_case=True) == "BCmodule"

    # Test combination of parameters
    config.length_sort = True
    config.force_to_top = {"priority"}
    result = module_key("priority", config, sub_imports=True)
    assert result.startswith("A") and "8:priority" in result


# LLM-generated content at query #22
#--------------------------

```python
def test_section_key():
    class MockConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Test basic section assignment
    config = MockConfig(force_to_top=["os"], lexicographical=False, length_sort=False)
    assert section_key("import os", config) == "Aos"
    assert section_key("import sys", config) == "Bsys"

    # Test lexicographical processing
    config = MockConfig(force_to_top=[], lexicographical=True, length_sort=False)
    assert section_key("from x import y", config) == "Bx.y"
    assert section_key("import x.y", config) == "Bx.y"

    # Test length sort
    config = MockConfig(force_to_top=[], lexicographical=False, length_sort=True)
    assert section_key("import a", config) == "B1a"
    assert section_key("import abc", config) == "B3abc"

    # Test group_by_package
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        group_by_package=True,
    )
    assert section_key("from package import module", config) == "Bpackage"

    # Test reverse_relative with sort_relative_in_force_sorted_sections=False
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        reverse_relative=True,
        sort_relative_in_force_sorted_sections=False,
    )
    assert section_key("from . import module", config) == "B. import module"

    # Test sort_relative_in_force_sorted_sections with reverse_relative
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        reverse_relative=True,
        sort_relative_in_force_sorted_sections=True,
    )
    assert section_key("from .. import module", config) == "B.. import module"

    # Test case sensitivity with honor_case_in_force_sorted_sections
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        case_sensitive=False,
        order_by_type=True,
        honor_case_in_force_sorted_sections=True,
    )
    assert section_key("import MyModule", config) == "Bmymodule"

    # Test order_by_type=False
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        case_sensitive=True,
        order_by_type=False,
        honor_case_in_force_sorted_sections=False,
    )
    assert section_key("import MyModule", config) == "Bmymodule"

    # Test complex import line
    config = MockConfig(force_to_top=[], lexicographical=True, length_sort=False)
    assert section_key("from a.b.c import d, e", config) == "Ba.b.c.d, e"

    # Test force_to_top with different sections
    config = MockConfig(force_to_top=["special"], lexicographical=False, length_sort=False)
    assert section_key("import special", config) == "Aspecial"
    assert section_key("import regular", config) == "Bregular"

    # Test mixed case handling with honor_case_in_force_sorted_sections
    config = MockConfig(
        force_to_top=[],
        lexicographical=False,
        length_sort=False,
        case_sensitive=True,
        order_by_type=False,
        honor_case_in_force_sorted_sections=True,
    )
    result = section_key("from MyPackage import MyClass", config)
    assert result == "Bmypackage import myclass"


# LLM-generated content at query #23
#--------------------------

```python
def test_module_key():
    from unittest.mock import Mock

    # Test basic module name without special handling
    config = Mock(
        reverse_relative=False,
        order_by_type=False,
        constants=set(),
        classes=set(),
        variables=set(),
        case_sensitive=True,
        length_sort=False,
        length_sort_straight=False,
        length_sort_sections=set(),
        force_to_top=set(),
    )
    assert module_key("os", config) == "Bos"

    # Test with relative imports and reverse_relative=True
    config.reverse_relative = True
    assert module_key("..module", config) == "B.. module"
    config.reverse_relative = False
    assert module_key("..module", config) == "B.._module"

    # Test ignore_case parameter
    assert module_key("OS", config, ignore_case=True) == "Bos"
    assert module_key("OS", config, ignore_case=False) == "BOS"

    # Test config.case_sensitive=False
    config.case_sensitive = False
    assert module_key("OS", config) == "Bos"
    config.case_sensitive = True

    # Test sub_imports with order_by_type
    config.order_by_type = True
    config.constants = {"CONSTANT"}
    config.classes = {"Class"}
    config.variables = {"variable"}

    assert module_key("CONSTANT", config, sub_imports=True) == "BACONSTANT"
    assert module_key("Class", config, sub_imports=True) == "BBClass"
    assert module_key("variable", config, sub_imports=True) == "BCvariable"
    assert module_key("Other", config, sub_imports=True) == "BBOther"
    assert module_key("other", config, sub_imports=True) == "BCother"

    # Test uppercase module name > 1 char (issue #376)
    assert module_key("ABC", config, sub_imports=True) == "BAABC"
    assert module_key("A", config, sub_imports=True) == "BCA"

    # Test length_sort
    config.length_sort = True
    assert module_key("long_module", config) == "B11:long_module"
    config.length_sort = False

    # Test length_sort_straight with straight_import
    config.length_sort_straight = True
    assert module_key("module", config, straight_import=True) == "B6:module"
    assert module_key("module", config, straight_import=False) == "Bmodule"
    config.length_sort_straight = False

    # Test length_sort_sections
    config.length_sort_sections = {"test_section"}
    assert module_key("module", config, section_name="test_section") == "B6:module"
    assert module_key("module", config, section_name="other_section") == "Bmodule"
    config.length_sort_sections = set()

    # Test force_to_top
    config.force_to_top = {"important"}
    assert module_key("important", config) == "Aimportant"
    assert module_key("regular", config) == "Bregular"
    config.force_to_top = set()

    # Test combined prefix and length sort
    config.order_by_type = True
    config.length_sort = True
    config.constants = {"MY_CONST"}
    result = module_key("MY_CONST", config, sub_imports=True)
    assert result.startswith("A8:MY_CONST") or result == "A8:MY_CONST"


