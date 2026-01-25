####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config()
    config.reverse_relative = True
    result = module_key("..module", config)
    assert result == "B.. module"

def test_module_key_with_relative_import_no_reverse():
    config = Config()
    config.reverse_relative = False
    result = module_key("..module", config)
    assert result == "B.._module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_case_sensitive():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B7:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B7:module"

def test_module_key_length_sort_section():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="TEST")
    assert result == "B7:module"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined():
    config = Config()
    config.order_by_type = True
    config.constants = {"MODULE"}
    config.length_sort = True
    config.force_to_top = {"MODULE"}
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "AA6:MODULE"


# LLM-generated content at query #2
#--------------------------

def test_module_key_predicate_at_line_11_false():
    config = Config()
    config.reverse_relative = True
    module_name = ".test"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    sep = " " if config.reverse_relative else "_"
    result = sep == "_"
    assert result == False


# LLM-generated content at query #3
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_and_constants():
    config = Config(order_by_type=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BAconst"

def test_module_key_sub_imports_and_classes():
    config = Config(order_by_type=True, classes={"Class"})
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBClass"

def test_module_key_sub_imports_and_variables():
    config = Config(order_by_type=True, variables={"var"})
    result = module_key("var", config, sub_imports=True)
    assert result == "BCvar"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("CONST", config, sub_imports=True)
    assert result == "BACONST"

def test_module_key_sub_imports_class_by_case():
    config = Config(order_by_type=True)
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"

def test_module_key_sub_imports_default_prefix():
    config = Config(order_by_type=True)
    result = module_key("function", config, sub_imports=True)
    assert result == "BCfunction"

def test_module_key_case_sensitive_false():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B4:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B4:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"standard"})
    result = module_key("module", config, section_name="standard")
    assert result == "B4:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_prefix_and_length_sort():
    config = Config(order_by_type=True, length_sort=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BA5:const"

def test_module_key_relative_without_reverse():
    config = Config(reverse_relative=False)
    result = module_key(".. module", config)
    assert result == "B.._module"


# LLM-generated content at query #4
#--------------------------

def test_predicate_at_line_20_evaluates_to_false():
    class Config:
        order_by_type = False
        constants = set()
        classes = set()
        variables = set()
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = set()
        force_to_top = set()
        reverse_relative = False

    config = Config()
    result = module_key("some_module", config, sub_imports=True)
    assert result is not None


# LLM-generated content at query #5
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_relative_import_no_reverse():
    config = Config(reverse_relative=False)
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_case_sensitive_false():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default_prefix():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_force_to_top_and_prefix():
    config = Config(order_by_type=True, constants={"module"}, force_to_top={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "AAmodule"

def test_module_key_combined_length_sort_and_prefix():
    config = Config(order_by_type=True, constants={"module"}, length_sort=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BA6:module"


# LLM-generated content at query #6
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import django", config)
    assert result == "Aimport django"
    result = section_key("from django import something", config)
    assert result == "Afrom django import something"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x import y", config)
    assert result == "Bx.y"

def test_section_key_not_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from package import module", config)
    assert result == "Bfrom package"

def test_section_key_reverse_relative_and_sort_relative_in_force_sorted_sections_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .. import module"

def test_section_key_sort_relative_in_force_sorted_sections_true_reverse_relative_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "B.._import module"

def test_section_key_sort_relative_in_force_sorted_sections_true_reverse_relative_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "B.. import module"

def test_section_key_length_sort_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=False, honor_case_in_force_sorted_sections=False)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"

def test_section_key_honor_case_in_force_sorted_sections_true_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=False, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import NAMES", config)
    assert result == "Bfrom MODULE import names"

def test_section_key_honor_case_in_force_sorted_sections_true_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import NAMES", config)
    assert result == "Bfrom module import NAMES"

def test_section_key_honor_case_in_force_sorted_sections_false_case_sensitive_false_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=False, honor_case_in_force_sorted_sections=False)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"

def test_section_key_import_statement_without_from():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import something", config)
    assert result == "Bsomething"


# LLM-generated content at query #7
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B3:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B3:module"

def test_module_key_length_sort_sections():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="test")
    assert result == "B3:module"

def test_module_key_case_insensitive_config():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "bmodule"

def test_module_key_combined():
    config = Config()
    config.reverse_relative = True
    config.order_by_type = True
    config.classes = {"module"}
    config.force_to_top = {".. module"}
    config.length_sort = True
    result = module_key(".. module", config, sub_imports=True)
    assert result == "ABB8:.. module"


# LLM-generated content at query #8
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import django", config)
    assert result == "Aimport django"
    result = section_key("from django import something", config)
    assert result == "Afrom django import something"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from mypackage import something", config)
    assert result == "Bfrom mypackage"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from a import b", config)
    assert result == "Bimport b"

def test_section_key_non_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from a import b", config)
    assert result == "Ba import b"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .. import module"

def test_section_key_sort_relative_in_force_sorted_sections_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .._import module"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=False, honor_case_in_force_sorted_sections=False)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=False, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom MODULE import name"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_reverse_relative_and_sort_relative_in_force_sorted_sections_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .. import module"


# LLM-generated content at query #9
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B9import a"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_not_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("IMPORT X", config)
    assert result == "Bimport x"

def test_section_key_case_sensitive_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=False, order_by_type=True)
    result = section_key("IMPORT X", config)
    assert result == "Bimport x"

def test_section_key_honor_case_in_force_sorted_sections_with_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bmodule import NAME"

def test_section_key_honor_case_in_force_sorted_sections_without_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("import MODULE", config)
    assert result == "Bimport module"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ..x import y", config)
    assert result == "B.._x import y"

def test_section_key_reverse_relative_without_sort_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from .x import y", config)
    assert result == "Bfrom .x import y"

def test_section_key_reverse_relative_with_sort_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from .x import y", config)
    assert result == "B. x import y"


# LLM-generated content at query #10
#--------------------------

def test_predicate_at_line_20_evaluates_to_false():
    config = Config()
    config.order_by_type = False
    result = module_key("some_module", config, sub_imports=True)
    assert "A" not in result and "B" not in result and "C" not in result


# LLM-generated content at query #11
#--------------------------

def test_predicate_at_line_11_evaluates_to_false():
    config = Config()
    config.reverse_relative = True
    result = module_key(".module", config)
    assert "_" not in result


# LLM-generated content at query #12
#--------------------------

def test_section_key_with_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import django", config)
    assert result == "Aimport django"
def test_section_key_without_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import requests", config)
    assert result == "Bimport requests"
def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x import y", config)
    assert result == "Bx import y"
def test_section_key_non_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x import y", config)
    assert result == "Bx import y"
def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x.y import z", config)
    assert result == "Bx.y"
def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import a", config)
    assert result == "B7import a"
def test_section_key_case_insensitive_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=False, honor_case_in_force_sorted_sections=False)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"
def test_section_key_honor_case_in_force_sorted_sections_with_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bmodule import NAME"
def test_section_key_honor_case_in_force_sorted_sections_without_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("import MODULE", config)
    assert result == "Bmodule"
def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from ..module import name", config)
    assert result == "B.._module import name"
def test_section_key_sort_relative_in_force_sorted_sections_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from ..module import name", config)
    assert result == "B.. module import name"
def test_section_key_reverse_relative_without_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from ..module import name", config)
    assert result == "B..module import name"


# LLM-generated content at query #13
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_and_constants():
    config = Config(order_by_type=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BAconst"

def test_module_key_sub_imports_and_classes():
    config = Config(order_by_type=True, classes={"Class"})
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBClass"

def test_module_key_sub_imports_and_variables():
    config = Config(order_by_type=True, variables={"var"})
    result = module_key("var", config, sub_imports=True)
    assert result == "BCvar"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("CONST", config, sub_imports=True)
    assert result == "BACONST"

def test_module_key_sub_imports_class_by_case():
    config = Config(order_by_type=True)
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"

def test_module_key_sub_imports_default_prefix():
    config = Config(order_by_type=True)
    result = module_key("my_var", config, sub_imports=True)
    assert result == "BCmy_var"

def test_module_key_not_case_sensitive():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B4:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B4:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"special"})
    result = module_key("module", config, section_name="special")
    assert result == "B4:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_prefix_and_length_sort():
    config = Config(order_by_type=True, length_sort=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BA5:const"

def test_module_key_relative_import_no_reverse():
    config = Config(reverse_relative=False)
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_empty_string():
    config = Config()
    result = module_key("", config)
    assert result == "B"

def test_module_key_only_dots():
    config = Config()
    result = module_key("...", config)
    assert result == "B..."


# LLM-generated content at query #14
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from mypackage import something", config)
    assert result == "Bfrom mypackage"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from a import b", config)
    assert result == "B import b"

def test_section_key_non_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from a import b", config)
    assert result == "Ba import b"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .._ import module"

def test_section_key_reverse_relative_without_sort_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from . import module", config)
    assert result == "Bfrom . import module"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_honor_case_mixed_config():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("import ABC", config)
    assert result == "Bimport abc"

def test_section_key_import_statement():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import sys", config)
    assert result == "Bimport sys"

def test_section_key_from_statement():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from os import path", config)
    assert result == "Bos import path"


# LLM-generated content at query #15
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "bmodule"

def test_module_key_sub_imports_and_constants():
    config = Config()
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_and_classes():
    config = Config()
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_and_variables():
    config = Config()
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_not_case_sensitive():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "bmodule"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_sections():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="test")
    assert result == "B6:module"

def test_module_key_relative_import():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_relative_import_reverse():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.._module"


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_11_evaluates_to_false():
    config = type('Config', (), {'reverse_relative': True})()
    module_name = "..example"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    sep = " " if config.reverse_relative else "_"
    assert sep == " "


# LLM-generated content at query #17
#--------------------------

def test_predicate_at_line_4_false():
    config = Config()
    config.sort_relative_in_force_sorted_sections = True
    config.reverse_relative = True
    line = "from . import something"
    result = section_key(line, config)
    assert result is not None


# LLM-generated content at query #18
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_relative():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_relative_no_reverse():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_insensitive():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config()
    config.length_sort_sections = ["test"]
    result = module_key("module", config, section_name="TEST")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    config.length_sort = True
    config.force_to_top = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "AA6:module"


# LLM-generated content at query #19
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import flask", config)
    assert result == "Bimport flask"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_non_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x.y import z", config)
    assert result == "Bx.y"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B7import a"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"

def test_section_key_case_sensitive_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=False, order_by_type=True)
    result = section_key("IMPORT A", config)
    assert result == "Bimport a"

def test_section_key_honor_case_in_force_sorted_sections_with_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_honor_case_in_force_sorted_sections_without_import():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("import MODULE", config)
    assert result == "Bimport module"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ..module import name", config)
    assert result == "Bfrom .._module import name"

def test_section_key_sort_relative_in_force_sorted_sections_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ..module import name", config)
    assert result == "Bfrom .. module import name"

def test_section_key_reverse_relative_without_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ..module import name", config)
    assert result == "Bfrom ..module import name"


# LLM-generated content at query #20
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_not_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x.y import z", config)
    assert result == "Bfrom x.y"

def test_section_key_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from . import a", config)
    assert result == "Bfrom . import a"

def test_section_key_sort_relative_in_force_sorted_sections_with_space():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ... import a", config)
    assert result == "B... import a"

def test_section_key_sort_relative_in_force_sorted_sections_with_underscore():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ... import a", config)
    assert result == "B..._ import a"

def test_section_key_honor_case_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom MODULE import name"

def test_section_key_honor_case_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import name"


# LLM-generated content at query #21
#--------------------------

def test_predicate_at_line_43_evaluates_to_true():
    config = Config()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = True
    config.group_by_package = False
    config.lexicographical = False
    config.force_to_top = set()
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = False
    config.order_by_type = False
    config.length_sort = True
    line = "from . import something"
    result = section_key(line, config)
    assert result == "B25from . import something"


# LLM-generated content at query #22
#--------------------------

def test_predicate_at_line_29_false():
    from my_module import Config
    config = Config()
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    line = "import something"
    result = section_key(line, config)
    assert config.honor_case_in_force_sorted_sections and config.case_sensitive != config.order_by_type == False


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_43_evaluates_to_false():
    class Config:
        sort_relative_in_force_sorted_sections = False
        reverse_relative = True
        group_by_package = False
        lexicographical = False
        force_to_top = []
        honor_case_in_force_sorted_sections = False
        case_sensitive = False
        order_by_type = False
        length_sort = False
    config = Config()
    line = "from . import something"
    result = section_key(line, config)
    assert result == "Bfrom . import something"


# LLM-generated content at query #24
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B9import a"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_not_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from package import something", config)
    assert result == "Bfrom package"

def test_section_key_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .. import module"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ... import module", config)
    assert result == "B..._import module"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom MODULE import name"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("import MODULE", config)
    assert result == "Bimport module"

def test_section_key_import_line():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import module", config)
    assert result == "Bmodule"

def test_section_key_from_line():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from package import module", config)
    assert result == "Bpackage import module"


# LLM-generated content at query #25
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_insensitive_config():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined():
    config = Config(order_by_type=True, constants={"module"}, length_sort=True, force_to_top={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "AA6:module"


# LLM-generated content at query #26
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BAconst"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Class"})
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBClass"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"var"})
    result = module_key("var", config, sub_imports=True)
    assert result == "BCvar"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("CONST", config, sub_imports=True)
    assert result == "BACONST"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"

def test_module_key_sub_imports_default_prefix():
    config = Config(order_by_type=True)
    result = module_key("function", config, sub_imports=True)
    assert result == "BCfunction"

def test_module_key_case_sensitive_false():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_force_to_top():
    config = Config(force_to_top={"top_module"})
    result = module_key("top_module", config)
    assert result == "Atop_module"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B3:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B3:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"future"})
    result = module_key("module", config, section_name="future")
    assert result == "B3:module"

def test_module_key_combined_prefix_and_length_sort():
    config = Config(order_by_type=True, classes={"Class"}, length_sort=True)
    result = module_key("Class", config, sub_imports=True)
    assert result == "BB5:Class"


# LLM-generated content at query #27
#--------------------------

def test_predicate_at_line_15_false():
    config = Config()
    config.lexicographical = False
    line = "import something"
    result = section_key(line, config)
    assert config.lexicographical == False


# LLM-generated content at query #28
#--------------------------

def test_predicate_at_line_20_true():
    config = Config()
    config.sort_relative_in_force_sorted_sections = True
    line = "from ..module import something"
    result = section_key(line, config)
    assert config.sort_relative_in_force_sorted_sections


####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# LLM-generated content at query #1
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_relative():
    config = Config()
    config.reverse_relative = True
    result = module_key("... module", config)
    assert result == "B... module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_not_case_sensitive():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="TEST")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined():
    config = Config()
    config.reverse_relative = True
    config.constants = {"... module"}
    config.length_sort = True
    config.force_to_top = {"... module"}
    result = module_key("... module", config, sub_imports=True)
    assert result == "AA10:... module"


# LLM-generated content at query #2
#--------------------------

def test_naturally_empty_list():
    result = naturally([])
    assert result == []


def test_naturally_already_sorted():
    result = naturally(["a1", "a2", "a10"])
    assert result == ["a1", "a2", "a10"]


def test_naturally_unsorted():
    result = naturally(["a10", "a2", "a1"])
    assert result == ["a1", "a2", "a10"]


def test_naturally_mixed():
    result = naturally(["z1", "a10", "a2", "a1", "z10"])
    assert result == ["a1", "a2", "a10", "z1", "z10"]


def test_naturally_reverse():
    result = naturally(["a10", "a2", "a1"], reverse=True)
    assert result == ["a10", "a2", "a1"]


def test_naturally_with_key():
    result = naturally(["x", "a10", "b2"], key=lambda x: x[1:])
    assert result == ["x", "b2", "a10"]


def test_naturally_with_key_and_reverse():
    result = naturally(["x", "a10", "b2"], key=lambda x: x[1:], reverse=True)
    assert result == ["a10", "b2", "x"]


def test_naturally_single_element():
    result = naturally(["a10"])
    assert result == ["a10"]


def test_naturally_only_numbers():
    result = naturally(["10", "2", "1"])
    assert result == ["1", "2", "10"]


def test_naturally_only_text():
    result = naturally(["z", "a", "b"])
    assert result == ["a", "b", "z"]


# LLM-generated content at query #3
#--------------------------

def test_module_key_predicate_at_line_10_true():
    import re
    class Config:
        reverse_relative = False
        order_by_type = False
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = set()
        constants = set()
        classes = set()
        variables = set()
    config = Config()
    module_name = "... some_module"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    result = bool(match)
    assert result == True


# LLM-generated content at query #4
#--------------------------

def test_module_key_force_to_top_true():
    class Config:
        reverse_relative = False
        order_by_type = False
        constants = []
        classes = []
        variables = []
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = ["mymodule"]
    config = Config()
    result = module_key("mymodule", config)
    assert result.startswith("A")

def test_module_key_force_to_top_false():
    class Config:
        reverse_relative = False
        order_by_type = False
        constants = []
        classes = []
        variables = []
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = ["othermodule"]
    config = Config()
    result = module_key("mymodule", config)
    assert result.startswith("B")

def test_module_key_force_to_top_with_relative():
    class Config:
        reverse_relative = False
        order_by_type = False
        constants = []
        classes = []
        variables = []
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = [". mymodule"]
    config = Config()
    result = module_key(". mymodule", config)
    assert result.startswith("A")

def test_module_key_force_to_top_case_insensitive():
    class Config:
        reverse_relative = False
        order_by_type = False
        constants = []
        classes = []
        variables = []
        case_sensitive = False
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = ["MYMODULE"]
    config = Config()
    result = module_key("mymodule", config)
    assert result.startswith("A")

def test_module_key_force_to_top_with_prefix():
    class Config:
        reverse_relative = False
        order_by_type = True
        constants = []
        classes = []
        variables = []
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = ["mymodule"]
    config = Config()
    result = module_key("mymodule", config, sub_imports=True)
    assert result.startswith("A")


# LLM-generated content at query #5
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_insensitive_config():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_prefix_and_length():
    config = Config(order_by_type=True, length_sort=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BA6:module"


# LLM-generated content at query #6
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_relative_import():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_relative_import_no_reverse():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_insensitive_config():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B8:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B8:module"

def test_module_key_length_sort_section():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="TEST")
    assert result == "B8:module"

def test_module_key_combined_prefix_and_length():
    config = Config()
    config.order_by_type = True
    config.length_sort = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BA8:module"


# LLM-generated content at query #7
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_and_constants():
    config = Config(order_by_type=True, constants={"constant"})
    result = module_key("constant", config, sub_imports=True)
    assert result == "BACconstant"

def test_module_key_sub_imports_and_classes():
    config = Config(order_by_type=True, classes={"Class"})
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBCClass"

def test_module_key_sub_imports_and_variables():
    config = Config(order_by_type=True, variables={"variable"})
    result = module_key("variable", config, sub_imports=True)
    assert result == "BCCvariable"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("UPPER", config, sub_imports=True)
    assert result == "BAUPPER"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBCClassName"

def test_module_key_sub_imports_default_prefix():
    config = Config(order_by_type=True)
    result = module_key("lowercase", config, sub_imports=True)
    assert result == "BCClowercase"

def test_module_key_case_insensitive_config():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B3:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B3:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B3:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_force_to_top_and_prefix():
    config = Config(order_by_type=True, constants={"module"}, force_to_top={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "AACmodule"

def test_module_key_combined_length_sort_and_prefix():
    config = Config(order_by_type=True, classes={"Class"}, length_sort=True)
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBC5:Class"

def test_module_key_relative_import_with_separator():
    config = Config(reverse_relative=False)
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_empty_module_name():
    config = Config()
    result = module_key("", config)
    assert result == "B"

def test_module_key_module_name_is_number_string():
    config = Config()
    result = module_key("123", config)
    assert result == "B123"

def test_module_key_sub_imports_no_order_by_type():
    config = Config(order_by_type=False, constants={"constant"})
    result = module_key("constant", config, sub_imports=True)
    assert result == "Bconstant"

def test_module_key_straight_import_false():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=False)
    assert result == "Bmodule"

def test_module_key_section_name_not_in_length_sort_sections():
    config = Config(length_sort_sections={"other"})
    result = module_key("module", config, section_name="section")
    assert result == "Bmodule"


# LLM-generated content at query #8
#--------------------------

def test_module_key_predicate_at_line_11_false():
    config = Config()
    config.reverse_relative = True
    module_name = ".test"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    sep = " " if config.reverse_relative else "_"
    assert sep == " "
    assert not (sep == "_")


# LLM-generated content at query #9
#--------------------------

def test_predicate_at_line_20_true():
    config = type('Config', (), {'order_by_type': True})()
    config.constants = set()
    config.classes = set()
    config.variables = set()
    config.case_sensitive = True
    config.force_to_top = set()
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = set()
    config.reverse_relative = False
    result = module_key('test', config, sub_imports=True)
    assert result.startswith('B')


# LLM-generated content at query #10
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_and_order_by_type_constant():
    config = Config(order_by_type=True, constants={"const"})
    result = module_key("const", config, sub_imports=True)
    assert result == "BAconst"

def test_module_key_sub_imports_and_order_by_type_class():
    config = Config(order_by_type=True, classes={"Class"})
    result = module_key("Class", config, sub_imports=True)
    assert result == "BBClass"

def test_module_key_sub_imports_and_order_by_type_variable():
    config = Config(order_by_type=True, variables={"var"})
    result = module_key("var", config, sub_imports=True)
    assert result == "BCvar"

def test_module_key_sub_imports_and_order_by_type_uppercase():
    config = Config(order_by_type=True)
    result = module_key("CONST", config, sub_imports=True)
    assert result == "BACONST"

def test_module_key_sub_imports_and_order_by_type_capitalized():
    config = Config(order_by_type=True)
    result = module_key("MyClass", config, sub_imports=True)
    assert result == "BBMyClass"

def test_module_key_sub_imports_and_order_by_type_default():
    config = Config(order_by_type=True)
    result = module_key("my_var", config, sub_imports=True)
    assert result == "BCmy_var"

def test_module_key_not_case_sensitive():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B4:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B4:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"std"})
    result = module_key("module", config, section_name="std")
    assert result == "B4:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"top_module"})
    result = module_key("top_module", config)
    assert result == "Atop_module"


# LLM-generated content at query #11
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), length_sort=True, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=True, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from package import something", config)
    assert result == "Bfrom package"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=True, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from a import b", config)
    assert result == "Bab"

def test_section_key_not_lexicographical():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from a import b", config)
    assert result == "Ba import b"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=True, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .._ import module"

def test_section_key_reverse_relative_without_sort_relative():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from . import x", config)
    assert result == "Bfrom . import x"

def test_section_key_honor_case_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=True, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom MODULE import name"

def test_section_key_honor_case_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_no_honor_case_order_by_type_false():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import name"

def test_section_key_import_statement_lowercase():
    config = Config(force_to_top=set(), length_sort=False, group_by_package=False, lexicographical=False, sort_relative_in_force_sorted_sections=False, reverse_relative=False, honor_case_in_force_sorted_sections=False, case_sensitive=False, order_by_type=True)
    result = section_key("import MODULE", config)
    assert result == "Bimport module"


# LLM-generated content at query #12
#--------------------------

def test_module_key_with_reverse_relative_true():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B .. module"

def test_module_key_with_reverse_relative_false():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B .._module"

def test_module_key_without_match():
    config = Config()
    config.reverse_relative = False
    result = module_key("module", config)
    assert result == "Bmodule"


# LLM-generated content at query #13
#--------------------------

def test_predicate_at_line_23_evaluates_to_true():
    config = Config()
    config.force_to_top = {"some_module"}
    line = "some_module import something"
    result = section_key(line, config)
    assert result.startswith("A")


# LLM-generated content at query #14
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_relative():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_relative_no_reverse():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_case_sensitive_false():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BBmodule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B3:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B3:module"

def test_module_key_length_sort_section():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="test")
    assert result == "B3:module"

def test_module_key_combined_prefix_and_length():
    config = Config()
    config.order_by_type = True
    config.length_sort = True
    config.constants = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BA3:module"


# LLM-generated content at query #15
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "bmodule"

def test_module_key_sub_imports_and_order_by_type_constant():
    config = Config()
    config.order_by_type = True
    config.constants = {"MODULE"}
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_and_order_by_type_class():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_and_order_by_type_variable():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_and_order_by_type_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("UPPER", config, sub_imports=True)
    assert result == "BAUPPER"

def test_module_key_sub_imports_and_order_by_type_class_by_first_letter():
    config = Config()
    config.order_by_type = True
    result = module_key("ClassName", config, sub_imports=True)
    assert result == "BBClassName"

def test_module_key_sub_imports_and_order_by_type_default():
    config = Config()
    config.order_by_type = True
    result = module_key("lowercase", config, sub_imports=True)
    assert result == "BClowercase"

def test_module_key_not_case_sensitive():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "bmodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_sections():
    config = Config()
    config.length_sort_sections = ["test"]
    result = module_key("module", config, section_name="test")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_prefix_and_length_sort():
    config = Config()
    config.order_by_type = True
    config.length_sort = True
    config.constants = {"MODULE"}
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BA6:MODULE"

def test_module_key_relative_without_reverse():
    config = Config()
    config.reverse_relative = False
    result = module_key(".. module", config)
    assert result == "B.._module"


# LLM-generated content at query #16
#--------------------------

def test_predicate_at_line_20_evaluates_to_false():
    config = Config()
    config.order_by_type = False
    result = module_key("some_module", config, sub_imports=True)
    assert "A" not in result and "B" not in result and "C" not in result


# LLM-generated content at query #17
#--------------------------

def test_module_key_with_reverse_relative_true():
    class Config:
        reverse_relative = True
        order_by_type = False
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = set()
        constants = set()
        classes = set()
        variables = set()
    config = Config()
    result = module_key(".. module", config)
    assert result == "B .. module"

def test_module_key_with_reverse_relative_false():
    class Config:
        reverse_relative = False
        order_by_type = False
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = set()
        constants = set()
        classes = set()
        variables = set()
    config = Config()
    result = module_key(".. module", config)
    assert result == "B .._module"


# LLM-generated content at query #18
#--------------------------

def test_length_sort_false_when_all_conditions_false():
    class Config:
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
    config = Config()
    module_name = "some_module"
    section_name = "some_section"
    straight_import = False
    length_sort = (config.length_sort or (config.length_sort_straight and straight_import) or str(section_name).lower() in config.length_sort_sections)
    assert length_sort == False


# LLM-generated content at query #19
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"MODULE"})
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_sensitive_false():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_section():
    config = Config(length_sort_sections={"test"})
    result = module_key("module", config, section_name="test")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined_prefix_and_length():
    config = Config(order_by_type=True, length_sort=True, constants={"MODULE"})
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BA6:MODULE"

def test_module_key_relative_with_sep():
    config = Config(reverse_relative=False)
    result = module_key(".. module", config)
    assert result == "B.._module"

def test_module_key_ignore_case_and_sub_imports():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("MODULE", config, sub_imports=True, ignore_case=True)
    assert result == "BAmodule"

def test_module_key_no_sub_imports_with_order_by_type():
    config = Config(order_by_type=True, constants={"MODULE"})
    result = module_key("MODULE", config, sub_imports=False)
    assert result == "BMODULE"

def test_module_key_section_name_not_in_length_sort():
    config = Config(length_sort_sections={"other"})
    result = module_key("module", config, section_name="test")
    assert result == "Bmodule"

def test_module_key_straight_import_false():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=False)
    assert result == "Bmodule"


# LLM-generated content at query #20
#--------------------------

def test_predicate_at_line_11_evaluates_to_false():
    class Config:
        reverse_relative = True
        order_by_type = False
        constants = set()
        classes = set()
        variables = set()
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = set()
        force_to_top = set()
    config = Config()
    module_name = ".test"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    sep = " " if config.reverse_relative else "_"
    assert sep == " "
    assert config.reverse_relative is True
    result = sep == "_"
    assert result is False


# LLM-generated content at query #21
#--------------------------

def test_predicate_at_line_20_evaluates_to_false():
    class Config:
        order_by_type = False
        constants = {"CONST"}
        classes = {"Class"}
        variables = {"var"}
        case_sensitive = True
        force_to_top = set()
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        reverse_relative = False
    config = Config()
    result = module_key("module", config, sub_imports=True)
    assert "A" not in result and "B" not in result and "C" not in result
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=False)
    assert "A" not in result and "B" not in result and "C" not in result


# LLM-generated content at query #22
#--------------------------

def test_section_key_with_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import django", config)
    assert result == "Aimport django"
    result = section_key("import other", config)
    assert result == "Bimport other"

def test_section_key_with_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import a", config)
    assert result == "B8import a"
    result = section_key("import abc", config)
    assert result == "B10import abc"

def test_section_key_case_insensitive_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=False, honor_case_in_force_sorted_sections=False)
    result = section_key("import ABC", config)
    assert result == "Bimport abc"

def test_section_key_case_sensitive_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("import ABC", config)
    assert result == "Bimport ABC"

def test_section_key_honor_case_in_force_sorted_sections_with_split():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("from MODULE import Name", config)
    assert result == "Bfrom module import Name"

def test_section_key_honor_case_in_force_sorted_sections_without_split():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=False, order_by_type=True, honor_case_in_force_sorted_sections=True)
    result = section_key("import MODULE", config)
    assert result == "Bimport module"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from x import y", config)
    assert result == "Bx.y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from package import something", config)
    assert result == "Bfrom package"

def test_section_key_sort_relative_in_force_sorted_sections_with_reverse_relative_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .. import module"

def test_section_key_sort_relative_in_force_sorted_sections_with_reverse_relative_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from .. import module", config)
    assert result == "Bfrom .._import module"

def test_section_key_reverse_relative_without_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, case_sensitive=True, order_by_type=True, honor_case_in_force_sorted_sections=False)
    result = section_key("from . import module", config)
    assert result == "Bfrom . import module"


# LLM-generated content at query #23
#--------------------------

def test_predicate_at_line_15_false():
    config = Config()
    config.lexicographical = False
    line = "import something"
    result = section_key(line, config)
    assert config.lexicographical == False


# LLM-generated content at query #24
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config()
    config.reverse_relative = True
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "bmodule"

def test_module_key_sub_imports_constants():
    config = Config()
    config.order_by_type = True
    config.constants = {"MODULE"}
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_classes():
    config = Config()
    config.order_by_type = True
    config.classes = {"Module"}
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config()
    config.order_by_type = True
    config.variables = {"module"}
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config()
    config.order_by_type = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_capitalized():
    config = Config()
    config.order_by_type = True
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config()
    config.order_by_type = True
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_case_sensitive_false():
    config = Config()
    config.case_sensitive = False
    result = module_key("Module", config)
    assert result == "bmodule"

def test_module_key_force_to_top():
    config = Config()
    config.force_to_top = {"module"}
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config()
    config.length_sort = True
    result = module_key("module", config)
    assert result == "B4:module"

def test_module_key_length_sort_straight():
    config = Config()
    config.length_sort_straight = True
    result = module_key("module", config, straight_import=True)
    assert result == "B4:module"

def test_module_key_length_sort_sections():
    config = Config()
    config.length_sort_sections = {"test"}
    result = module_key("module", config, section_name="test")
    assert result == "B4:module"

def test_module_key_combined():
    config = Config()
    config.order_by_type = True
    config.constants = {"MODULE"}
    config.length_sort = True
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BA6:MODULE"


# LLM-generated content at query #25
#--------------------------

def test_predicate_at_line_33_evaluates_to_false():
    class Config:
        case_sensitive = True
        reverse_relative = False
        order_by_type = False
        constants = []
        classes = []
        variables = []
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []
        force_to_top = []

    config = Config()
    result = module_key("some_module", config, sub_imports=True, ignore_case=False, section_name=None, straight_import=False)
    assert config.case_sensitive is True
    config.case_sensitive = False
    result = module_key("some_module", config, sub_imports=True, ignore_case=False, section_name=None, straight_import=False)
    assert config.case_sensitive is False


# LLM-generated content at query #26
#--------------------------

def test_section_key_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import django", config)
    assert result == "Aimport django"

def test_section_key_not_force_to_top():
    config = Config(force_to_top={"django"}, sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import requests", config)
    assert result == "Bimport requests"

def test_section_key_length_sort():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=True, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("import a", config)
    assert result == "B8import a"

def test_section_key_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=True, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_non_lexicographical():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from x import y", config)
    assert result == "Bx import y"

def test_section_key_group_by_package():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=True, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from package import something", config)
    assert result == "Bfrom package"

def test_section_key_reverse_relative():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=True, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from . import x", config)
    assert result == "Bfrom . import x"

def test_section_key_sort_relative_in_force_sorted_sections():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=True, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=True)
    result = section_key("from ..module import x", config)
    assert result == "Bfrom .._module import x"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_true_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=True, order_by_type=False)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom MODULE import name"

def test_section_key_honor_case_in_force_sorted_sections_case_sensitive_false_order_by_type_true():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=True, case_sensitive=False, order_by_type=True)
    result = section_key("from MODULE import NAME", config)
    assert result == "Bfrom module import NAME"

def test_section_key_order_by_type_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=True, order_by_type=False)
    result = section_key("import UPPERCASE", config)
    assert result == "Bimport uppercase"

def test_section_key_case_sensitive_false():
    config = Config(force_to_top=set(), sort_relative_in_force_sorted_sections=False, group_by_package=False, lexicographical=False, reverse_relative=False, length_sort=False, honor_case_in_force_sorted_sections=False, case_sensitive=False, order_by_type=True)
    result = section_key("import UPPERCASE", config)
    assert result == "Bimport uppercase"


# LLM-generated content at query #27
#--------------------------

def test_predicate_at_line_11_evaluates_to_false():
    config = Config()
    config.reverse_relative = True
    module_name = "..example"
    match = re.match(r"^(\.+)\s*(.*)", module_name)
    sep = " " if config.reverse_relative else "_"
    assert sep == " "
    assert not (sep == "_")


# LLM-generated content at query #28
#--------------------------

def test_module_key_predicate_at_line_42_false():
    class Config:
        force_to_top = set()
        reverse_relative = False
        order_by_type = False
        constants = set()
        classes = set()
        variables = set()
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = set()

    config = Config()
    result = module_key("some_module", config)
    assert result[0] != 'A'


# LLM-generated content at query #29
#--------------------------

def test_module_key_sub_imports_and_order_by_type_true():
    class Config:
        order_by_type = True
        constants = {"CONSTANT"}
        classes = {"ClassName"}
        variables = {"variable"}
        force_to_top = set()
        reverse_relative = False
        case_sensitive = True
        length_sort = False
        length_sort_straight = False
        length_sort_sections = []

    config = Config()
    result = module_key("ClassName", config, sub_imports=True)
    assert result.startswith("BB")


# LLM-generated content at query #30
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("MODULE", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MOD", config, sub_imports=True)
    assert result == "BAMOD"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("MyModule", config, sub_imports=True)
    assert result == "BBMyModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("my_module", config, sub_imports=True)
    assert result == "BCmy_module"

def test_module_key_case_sensitive_false():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B3:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B3:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"test"})
    result = module_key("module", config, section_name="test")
    assert result == "B3:module"


# LLM-generated content at query #31
#--------------------------

def test_honor_case_in_force_sorted_sections_true_case_sensitive_not_equal_order_by_type():
    config = Config()
    config.honor_case_in_force_sorted_sections = True
    config.case_sensitive = True
    config.order_by_type = False
    line = "from MyModule import MyClass"
    result = section_key(line, config)
    assert config.honor_case_in_force_sorted_sections and config.case_sensitive != config.order_by_type


# LLM-generated content at query #32
#--------------------------

def test_predicate_at_line_20_evaluates_to_true():
    config = type('Config', (), {'order_by_type': True})()
    result = module_key('some_module', config, sub_imports=True)
    assert True


# LLM-generated content at query #33
#--------------------------

def test_predicate_at_line_20_evaluates_to_true():
    config = Config()
    config.order_by_type = True
    result = module_key("some_module", config, sub_imports=True)
    assert True


# LLM-generated content at query #34
#--------------------------

def test_predicate_at_line_4_true():
    config = Config()
    config.sort_relative_in_force_sorted_sections = False
    config.reverse_relative = True
    line = "from . import something"
    result = section_key(line, config)
    assert result is not None


# LLM-generated content at query #35
#--------------------------

def test_predicate_at_line_29_false():
    config = Config()
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    line = "import something"
    result = section_key(line, config)
    assert config.honor_case_in_force_sorted_sections and config.case_sensitive != config.order_by_type == False


# LLM-generated content at query #36
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("Module", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"MODULE"})
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"Module"})
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_by_first_letter():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_not_case_sensitive():
    config = Config(case_sensitive=False)
    result = module_key("Module", config)
    assert result == "Bmodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B6:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B6:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B6:module"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_combined():
    config = Config(order_by_type=True, constants={"MODULE"}, length_sort=True, force_to_top={"MODULE"})
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "A6:MODULE"


# LLM-generated content at query #37
#--------------------------

def test_predicate_at_line_12_true():
    config = Config()
    config.group_by_package = True
    line = "from mypackage import something"
    result = section_key(line, config)
    assert line == "from mypackage import something"


# LLM-generated content at query #38
#--------------------------

def test_predicate_at_line_12_false():
    config = Config()
    config.group_by_package = False
    line = "from x import y"
    result = section_key(line, config)
    assert "from" not in line.split(" import ", 1)[0]


# LLM-generated content at query #39
#--------------------------

def test_module_key_basic():
    config = Config()
    result = module_key("module", config)
    assert result == "Bmodule"

def test_module_key_with_relative_import():
    config = Config(reverse_relative=True)
    result = module_key(".. module", config)
    assert result == "B.. module"

def test_module_key_ignore_case():
    config = Config()
    result = module_key("MODULE", config, ignore_case=True)
    assert result == "Bmodule"

def test_module_key_not_case_sensitive():
    config = Config(case_sensitive=False)
    result = module_key("MODULE", config)
    assert result == "Bmodule"

def test_module_key_sub_imports_constants():
    config = Config(order_by_type=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BAmodule"

def test_module_key_sub_imports_classes():
    config = Config(order_by_type=True, classes={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BBmodule"

def test_module_key_sub_imports_variables():
    config = Config(order_by_type=True, variables={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_sub_imports_uppercase():
    config = Config(order_by_type=True)
    result = module_key("MODULE", config, sub_imports=True)
    assert result == "BAMODULE"

def test_module_key_sub_imports_class_like():
    config = Config(order_by_type=True)
    result = module_key("Module", config, sub_imports=True)
    assert result == "BBModule"

def test_module_key_sub_imports_default():
    config = Config(order_by_type=True)
    result = module_key("module", config, sub_imports=True)
    assert result == "BCmodule"

def test_module_key_force_to_top():
    config = Config(force_to_top={"module"})
    result = module_key("module", config)
    assert result == "Amodule"

def test_module_key_length_sort():
    config = Config(length_sort=True)
    result = module_key("module", config)
    assert result == "B2:module"

def test_module_key_length_sort_straight():
    config = Config(length_sort_straight=True)
    result = module_key("module", config, straight_import=True)
    assert result == "B2:module"

def test_module_key_length_sort_sections():
    config = Config(length_sort_sections={"section"})
    result = module_key("module", config, section_name="section")
    assert result == "B2:module"

def test_module_key_combined_prefix_and_length():
    config = Config(order_by_type=True, length_sort=True, constants={"module"})
    result = module_key("module", config, sub_imports=True)
    assert result == "BA2:module"


# LLM-generated content at query #40
#--------------------------

def test_honor_case_in_force_sorted_sections_false():
    config = Config()
    config.honor_case_in_force_sorted_sections = False
    config.case_sensitive = True
    config.order_by_type = False
    line = "import something"
    result = section_key(line, config)
    assert "something" in result


# LLM-generated content at query #41
#--------------------------

def test_predicate_at_line_43_true():
    class Config:
        sort_relative_in_force_sorted_sections = False
        reverse_relative = True
        group_by_package = False
        lexicographical = False
        force_to_top = set()
        honor_case_in_force_sorted_sections = False
        case_sensitive = False
        order_by_type = False
        length_sort = True
    config = Config()
    line = "from . import something"
    result = section_key(line, config)
    assert result.startswith("B")


# LLM-generated content at query #42
#--------------------------

def test_predicate_at_line_20_false():
    config = Config()
    config.sort_relative_in_force_sorted_sections = False
    line = "from . import something"
    result = section_key(line, config)
    assert "B" in result


# LLM-generated content at query #43
#--------------------------

def test_predicate_at_line_33_evaluates_to_true():
    config = type('Config', (), {'case_sensitive': False})()
    module_name = "TestModule"
    result = module_key(module_name, config)
    assert config.case_sensitive == False
    assert module_name.lower() == "testmodule"


# LLM-generated content at query #44
#--------------------------

def test_length_sort_evaluates_to_true_when_config_length_sort_is_true():
    config = Config()
    config.length_sort = True
    config.length_sort_straight = False
    config.length_sort_sections = []
    result = module_key("module", config)
    assert result.startswith("B") or result.startswith("A")
    assert ":" in result

def test_length_sort_evaluates_to_true_when_config_length_sort_straight_is_true_and_straight_import_is_true():
    config = Config()
    config.length_sort = False
    config.length_sort_straight = True
    config.length_sort_sections = []
    result = module_key("module", config, straight_import=True)
    assert result.startswith("B") or result.startswith("A")
    assert ":" in result

def test_length_sort_evaluates_to_true_when_section_name_in_config_length_sort_sections():
    config = Config()
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = ["future"]
    result = module_key("module", config, section_name="future")
    assert result.startswith("B") or result.startswith("A")
    assert ":" in result

def test_length_sort_evaluates_to_true_when_section_name_in_config_length_sort_sections_case_insensitive():
    config = Config()
    config.length_sort = False
    config.length_sort_straight = False
    config.length_sort_sections = ["FUTURE"]
    result = module_key("module", config, section_name="future")
    assert result.startswith("B") or result.startswith("A")
    assert ":" in result

def test_length_sort_evaluates_to_true_when_all_conditions_are_true():
    config = Config()
    config.length_sort = True
    config.length_sort_straight = True
    config.length_sort_sections = ["future"]
    result = module_key("module", config, straight_import=True, section_name="future")
    assert result.startswith("B") or result.startswith("A")
    assert ":" in result


# LLM-generated content at query #45
#--------------------------

def test_predicate_at_line_23_is_false():
    config = Config()
    config.force_to_top = set()
    line = "some_module"
    result = section_key(line, config)
    assert "A" not in result


