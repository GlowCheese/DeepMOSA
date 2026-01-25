# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import urllib.request as module_1
import isort.wrap_modes as module_2
import genericpath as module_3
import isort.settings as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0, raise_on_skip=var_0)

def test_case_1():
    pass

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.thishost()
    assert module_1.MAXFTPCACHE == 10
    assert module_1.ftpcache == {}
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '"['
    module_0.process(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = "C'aC*Jqn!\r"
    var_1 = [var_0]
    var_2 = None
    module_0.process(var_1, var_2)

def test_case_5():
    var_0 = []
    var_1 = module_0.process(var_0, var_0, raise_on_skip=var_0)
    assert var_1 is False
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\r/Yi<.&'
    module_0.process(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '# isort: off\nimport sys\nimp!rt os\n'
    var_1 = [var_0]
    module_0.process(var_1, var_1)

def test_case_8():
    var_0 = 'from syimport aIvefoos ort aF\n'
    var_1 = [var_0, var_0]
    var_2 = module_0.process(var_1, var_1, var_0)
    assert var_2 is False
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'from sy import argv\nfrom os mort pat\n'
    var_1 = [var_0, var_0]
    module_0.process(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'from sy import arv\nfr os mort pa'
    var_1 = [var_0]
    module_0.process(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'from sy impo+t arv\nfr os mort pa'
    var_1 = [var_0]
    module_0.process(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '3b\n\'sch2\x0b"v)KF+vaK'
    var_1 = [var_0]
    module_0.process(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = '+9"Xi@\r"78y59;yn\''
    var_1 = [var_0, var_0, var_0]
    var_2 = None
    module_0.process(var_1, var_2)

def test_case_14():
    var_0 = 'from syimport avefos ort aF\n'
    var_1 = [var_0, var_0, var_0, var_0, var_0]
    var_2 = module_0.process(var_1, var_1, var_0)
    assert var_2 is False
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}
    var_3 = module_3.getctime(var_2)
    assert var_3 == pytest.approx(1768772393.84946, abs=0.01, rel=0.01)
    assert f'{type(module_3.ALLOW_MISSING).__module__}.{type(module_3.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'

def test_case_15():
    var_0 = 88
    var_1 = 2
    var_2 = 'line_length'
    var_3 = 'wrap_length'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_4.Config(**var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.settings.Config'
    assert var_5.py_version == 'py3'
    assert f'{type(var_5.force_to_top).__module__}.{type(var_5.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.force_to_top) == 0
    assert f'{type(var_5.skip).__module__}.{type(var_5.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.skip) == 19
    assert f'{type(var_5.extend_skip).__module__}.{type(var_5.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extend_skip) == 0
    assert f'{type(var_5.skip_glob).__module__}.{type(var_5.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.skip_glob) == 0
    assert f'{type(var_5.extend_skip_glob).__module__}.{type(var_5.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extend_skip_glob) == 0
    assert var_5.skip_gitignore is False
    assert var_5.line_length == 88
    assert var_5.wrap_length == 2
    assert var_5.line_ending == ''
    assert var_5.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_5.no_sections is False
    assert f'{type(var_5.known_future_library).__module__}.{type(var_5.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_future_library) == 1
    assert f'{type(var_5.known_third_party).__module__}.{type(var_5.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_third_party) == 0
    assert f'{type(var_5.known_first_party).__module__}.{type(var_5.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_first_party) == 0
    assert f'{type(var_5.known_local_folder).__module__}.{type(var_5.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_local_folder) == 0
    assert f'{type(var_5.known_standard_library).__module__}.{type(var_5.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.known_standard_library) == 234
    assert f'{type(var_5.extra_standard_library).__module__}.{type(var_5.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.extra_standard_library) == 0
    assert var_5.known_other == {}
    assert var_5.multi_line_output == module_2.WrapModes.GRID
    assert var_5.forced_separate == ()
    assert var_5.indent == '    '
    assert var_5.comment_prefix == '  #'
    assert var_5.length_sort is False
    assert var_5.length_sort_straight is False
    assert f'{type(var_5.length_sort_sections).__module__}.{type(var_5.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.length_sort_sections) == 0
    assert f'{type(var_5.add_imports).__module__}.{type(var_5.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.add_imports) == 0
    assert f'{type(var_5.remove_imports).__module__}.{type(var_5.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.remove_imports) == 0
    assert var_5.append_only is False
    assert var_5.reverse_relative is False
    assert var_5.force_single_line is False
    assert var_5.single_line_exclusions == ()
    assert var_5.default_section == 'THIRDPARTY'
    assert var_5.import_headings == {}
    assert var_5.import_footers == {}
    assert var_5.balanced_wrapping is False
    assert var_5.use_parentheses is False
    assert var_5.order_by_type is True
    assert var_5.atomic is False
    assert var_5.lines_before_imports == -1
    assert var_5.lines_after_imports == -1
    assert var_5.lines_between_sections == 1
    assert var_5.lines_between_types == 0
    assert var_5.combine_as_imports is False
    assert var_5.combine_star is False
    assert var_5.include_trailing_comma is False
    assert var_5.from_first is False
    assert var_5.verbose is False
    assert var_5.quiet is False
    assert var_5.force_adds is False
    assert var_5.force_alphabetical_sort_within_sections is False
    assert var_5.force_alphabetical_sort is False
    assert var_5.force_grid_wrap == 0
    assert var_5.force_sort_within_sections is False
    assert var_5.lexicographical is False
    assert var_5.group_by_package is False
    assert var_5.ignore_whitespace is False
    assert f'{type(var_5.no_lines_before).__module__}.{type(var_5.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.no_lines_before) == 0
    assert var_5.no_inline_sort is False
    assert var_5.ignore_comments is False
    assert var_5.case_sensitive is False
    assert f'{type(var_5.sources).__module__}.{type(var_5.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_5.sources) == 2
    assert var_5.virtual_env == ''
    assert var_5.conda_env == ''
    assert var_5.ensure_newline_before_comments is False
    assert var_5.directory == '/workspace'
    assert var_5.profile == ''
    assert var_5.honor_noqa is False
    assert f'{type(var_5.src_paths).__module__}.{type(var_5.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_5.src_paths) == 2
    assert var_5.remove_redundant_aliases is False
    assert var_5.float_to_top is False
    assert var_5.filter_files is False
    assert var_5.formatter == ''
    assert var_5.formatting_function is None
    assert var_5.color_output is False
    assert f'{type(var_5.treat_comments_as_code).__module__}.{type(var_5.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.treat_comments_as_code) == 0
    assert var_5.treat_all_comments_as_code is False
    assert f'{type(var_5.supported_extensions).__module__}.{type(var_5.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.supported_extensions) == 4
    assert f'{type(var_5.blocked_extensions).__module__}.{type(var_5.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.blocked_extensions) == 1
    assert f'{type(var_5.constants).__module__}.{type(var_5.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.constants) == 0
    assert f'{type(var_5.classes).__module__}.{type(var_5.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.classes) == 0
    assert f'{type(var_5.variables).__module__}.{type(var_5.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.variables) == 0
    assert var_5.dedup_headings is False
    assert var_5.only_sections is False
    assert var_5.only_modified is False
    assert var_5.combine_straight_imports is False
    assert var_5.auto_identify_namespace_packages is True
    assert f'{type(var_5.namespace_packages).__module__}.{type(var_5.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_5.namespace_packages) == 0
    assert var_5.follow_links is True
    assert var_5.indented_import_headings is True
    assert var_5.honor_case_in_force_sorted_sections is False
    assert var_5.sort_relative_in_force_sorted_sections is False
    assert var_5.overwrite_in_place is False
    assert var_5.reverse_sort is False
    assert var_5.star_first is False
    assert var_5.git_ls_files == {}
    assert var_5.format_error == '{error}: {message}'
    assert var_5.format_success == '{success}: {message}'
    assert var_5.sort_order == 'natural'
    assert var_5.sort_reexports is False
    assert var_5.split_on_trailing_comma is False
    assert module_4.TYPE_CHECKING is False
    assert module_4.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_4.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.FIRSTPARTY == 'FIRSTPARTY'
    assert module_4.FUTURE == 'FUTURE'
    assert module_4.LOCALFOLDER == 'LOCALFOLDER'
    assert module_4.STDLIB == 'STDLIB'
    assert module_4.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_4.CYTHON_EXTENSIONS).__module__}.{type(module_4.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_4.SUPPORTED_EXTENSIONS).__module__}.{type(module_4.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_4.BLOCKED_EXTENSIONS).__module__}.{type(module_4.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.BLOCKED_EXTENSIONS) == 1
    assert module_4.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_4.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_4.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_4.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_4.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_4.DEFAULT_SKIP).__module__}.{type(module_4.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_SKIP) == 19
    assert module_4.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_4.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_4.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_4.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_4.KNOWN_PREFIX == 'known_'
    assert module_4.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_4.RUNTIME_SOURCE == 'runtime'
    assert module_4.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_4.DEFAULT_CONFIG).__module__}.{type(module_4.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_4.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_4.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_4.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip).__module__}.{type(module_4.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_4.DEFAULT_CONFIG.skip_gitignore is False
    assert module_4.DEFAULT_CONFIG.line_length == 79
    assert module_4.DEFAULT_CONFIG.wrap_length == 0
    assert module_4.DEFAULT_CONFIG.line_ending == ''
    assert module_4.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_4.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_4.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_4.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_4.DEFAULT_CONFIG.known_other == {}
    assert module_4.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_4.DEFAULT_CONFIG.forced_separate == ()
    assert module_4.DEFAULT_CONFIG.indent == '    '
    assert module_4.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_4.DEFAULT_CONFIG.length_sort is False
    assert module_4.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_4.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_4.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.add_imports).__module__}.{type(module_4.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_4.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.remove_imports) == 0
    assert module_4.DEFAULT_CONFIG.append_only is False
    assert module_4.DEFAULT_CONFIG.reverse_relative is False
    assert module_4.DEFAULT_CONFIG.force_single_line is False
    assert module_4.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_4.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_4.DEFAULT_CONFIG.import_headings == {}
    assert module_4.DEFAULT_CONFIG.import_footers == {}
    assert module_4.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_4.DEFAULT_CONFIG.use_parentheses is False
    assert module_4.DEFAULT_CONFIG.order_by_type is True
    assert module_4.DEFAULT_CONFIG.atomic is False
    assert module_4.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_4.DEFAULT_CONFIG.lines_between_types == 0
    assert module_4.DEFAULT_CONFIG.combine_as_imports is False
    assert module_4.DEFAULT_CONFIG.combine_star is False
    assert module_4.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_4.DEFAULT_CONFIG.from_first is False
    assert module_4.DEFAULT_CONFIG.verbose is False
    assert module_4.DEFAULT_CONFIG.quiet is False
    assert module_4.DEFAULT_CONFIG.force_adds is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_4.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_4.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.lexicographical is False
    assert module_4.DEFAULT_CONFIG.group_by_package is False
    assert module_4.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_4.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_4.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_4.DEFAULT_CONFIG.no_inline_sort is False
    assert module_4.DEFAULT_CONFIG.ignore_comments is False
    assert module_4.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_4.DEFAULT_CONFIG.sources).__module__}.{type(module_4.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.sources) == 1
    assert module_4.DEFAULT_CONFIG.virtual_env == ''
    assert module_4.DEFAULT_CONFIG.conda_env == ''
    assert module_4.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_4.DEFAULT_CONFIG.directory == '/workspace'
    assert module_4.DEFAULT_CONFIG.profile == ''
    assert module_4.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_4.DEFAULT_CONFIG.src_paths).__module__}.{type(module_4.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.src_paths) == 2
    assert module_4.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_4.DEFAULT_CONFIG.float_to_top is False
    assert module_4.DEFAULT_CONFIG.filter_files is False
    assert module_4.DEFAULT_CONFIG.formatter == ''
    assert module_4.DEFAULT_CONFIG.formatting_function is None
    assert module_4.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_4.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_4.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_4.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.constants).__module__}.{type(module_4.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.classes).__module__}.{type(module_4.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.variables).__module__}.{type(module_4.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.variables) == 0
    assert module_4.DEFAULT_CONFIG.dedup_headings is False
    assert module_4.DEFAULT_CONFIG.only_sections is False
    assert module_4.DEFAULT_CONFIG.only_modified is False
    assert module_4.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_4.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_4.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_4.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_4.DEFAULT_CONFIG.follow_links is True
    assert module_4.DEFAULT_CONFIG.indented_import_headings is True
    assert module_4.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_4.DEFAULT_CONFIG.reverse_sort is False
    assert module_4.DEFAULT_CONFIG.star_first is False
    assert module_4.DEFAULT_CONFIG.git_ls_files == {}
    assert module_4.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_4.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_4.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_4.DEFAULT_CONFIG.sort_reexports is False
    assert module_4.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_4.Config.known_patterns).__module__}.{type(module_4.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments).__module__}.{type(module_4.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments_end).__module__}.{type(module_4.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skips).__module__}.{type(module_4.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skip_globs).__module__}.{type(module_4.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.sorting_function).__module__}.{type(module_4.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_6 = '    '
    var_7 = module_0._indented_config(var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.settings.Config'
    assert var_7.py_version == 'py3'
    assert f'{type(var_7.force_to_top).__module__}.{type(var_7.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.force_to_top) == 0
    assert f'{type(var_7.skip).__module__}.{type(var_7.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip) == 19
    assert f'{type(var_7.extend_skip).__module__}.{type(var_7.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip) == 0
    assert f'{type(var_7.skip_glob).__module__}.{type(var_7.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.skip_glob) == 0
    assert f'{type(var_7.extend_skip_glob).__module__}.{type(var_7.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extend_skip_glob) == 0
    assert var_7.skip_gitignore is False
    assert var_7.line_length == 84
    assert var_7.wrap_length == 0
    assert var_7.line_ending == ''
    assert var_7.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_7.no_sections is False
    assert f'{type(var_7.known_future_library).__module__}.{type(var_7.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_future_library) == 1
    assert f'{type(var_7.known_third_party).__module__}.{type(var_7.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_third_party) == 0
    assert f'{type(var_7.known_first_party).__module__}.{type(var_7.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_first_party) == 0
    assert f'{type(var_7.known_local_folder).__module__}.{type(var_7.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_local_folder) == 0
    assert f'{type(var_7.known_standard_library).__module__}.{type(var_7.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.known_standard_library) == 234
    assert f'{type(var_7.extra_standard_library).__module__}.{type(var_7.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.extra_standard_library) == 0
    assert var_7.known_other == {}
    assert var_7.multi_line_output == module_2.WrapModes.GRID
    assert var_7.forced_separate == ()
    assert var_7.indent == '    '
    assert var_7.comment_prefix == '  #'
    assert var_7.length_sort is False
    assert var_7.length_sort_straight is False
    assert f'{type(var_7.length_sort_sections).__module__}.{type(var_7.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.length_sort_sections) == 0
    assert f'{type(var_7.add_imports).__module__}.{type(var_7.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.add_imports) == 0
    assert f'{type(var_7.remove_imports).__module__}.{type(var_7.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.remove_imports) == 0
    assert var_7.append_only is False
    assert var_7.reverse_relative is False
    assert var_7.force_single_line is False
    assert var_7.single_line_exclusions == ()
    assert var_7.default_section == 'THIRDPARTY'
    assert var_7.import_headings == {}
    assert var_7.import_footers == {}
    assert var_7.balanced_wrapping is False
    assert var_7.use_parentheses is False
    assert var_7.order_by_type is True
    assert var_7.atomic is False
    assert var_7.lines_before_imports == -1
    assert var_7.lines_after_imports == 1
    assert var_7.lines_between_sections == 1
    assert var_7.lines_between_types == 0
    assert var_7.combine_as_imports is False
    assert var_7.combine_star is False
    assert var_7.include_trailing_comma is False
    assert var_7.from_first is False
    assert var_7.verbose is False
    assert var_7.quiet is False
    assert var_7.force_adds is False
    assert var_7.force_alphabetical_sort_within_sections is False
    assert var_7.force_alphabetical_sort is False
    assert var_7.force_grid_wrap == 0
    assert var_7.force_sort_within_sections is False
    assert var_7.lexicographical is False
    assert var_7.group_by_package is False
    assert var_7.ignore_whitespace is False
    assert f'{type(var_7.no_lines_before).__module__}.{type(var_7.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.no_lines_before) == 0
    assert var_7.no_inline_sort is False
    assert var_7.ignore_comments is False
    assert var_7.case_sensitive is False
    assert f'{type(var_7.sources).__module__}.{type(var_7.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_7.sources) == 2
    assert var_7.virtual_env == ''
    assert var_7.conda_env == ''
    assert var_7.ensure_newline_before_comments is False
    assert var_7.directory == '/workspace'
    assert var_7.profile == ''
    assert var_7.honor_noqa is False
    assert f'{type(var_7.src_paths).__module__}.{type(var_7.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_7.src_paths) == 2
    assert var_7.remove_redundant_aliases is False
    assert var_7.float_to_top is False
    assert var_7.filter_files is False
    assert var_7.formatter == ''
    assert var_7.formatting_function is None
    assert var_7.color_output is False
    assert f'{type(var_7.treat_comments_as_code).__module__}.{type(var_7.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.treat_comments_as_code) == 0
    assert var_7.treat_all_comments_as_code is False
    assert f'{type(var_7.supported_extensions).__module__}.{type(var_7.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.supported_extensions) == 4
    assert f'{type(var_7.blocked_extensions).__module__}.{type(var_7.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.blocked_extensions) == 1
    assert f'{type(var_7.constants).__module__}.{type(var_7.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.constants) == 0
    assert f'{type(var_7.classes).__module__}.{type(var_7.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.classes) == 0
    assert f'{type(var_7.variables).__module__}.{type(var_7.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.variables) == 0
    assert var_7.dedup_headings is False
    assert var_7.only_sections is False
    assert var_7.only_modified is False
    assert var_7.combine_straight_imports is False
    assert var_7.auto_identify_namespace_packages is True
    assert f'{type(var_7.namespace_packages).__module__}.{type(var_7.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_7.namespace_packages) == 0
    assert var_7.follow_links is True
    assert var_7.indented_import_headings is True
    assert var_7.honor_case_in_force_sorted_sections is False
    assert var_7.sort_relative_in_force_sorted_sections is False
    assert var_7.overwrite_in_place is False
    assert var_7.reverse_sort is False
    assert var_7.star_first is False
    assert var_7.git_ls_files == {}
    assert var_7.format_error == '{error}: {message}'
    assert var_7.format_success == '{success}: {message}'
    assert var_7.sort_order == 'natural'
    assert var_7.sort_reexports is False
    assert var_7.split_on_trailing_comma is False
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}
    var_8 = var_7.wrap_length
    assert var_8 == 0

def test_case_16():
    var_0 = 'FUTURE'
    var_1 = '# Future imports'
    var_2 = {var_0: var_1}
    var_3 = 88
    var_4 = 79
    var_5 = False
    var_6 = 'line_length'
    var_7 = 'wrap_length'
    var_8 = 'import_headings'
    var_9 = 'indented_import_headings'
    var_10 = {var_6: var_3, var_7: var_4, var_8: var_2, var_9: var_5}
    var_11 = module_4.Config(**var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.settings.Config'
    assert var_11.py_version == 'py3'
    assert f'{type(var_11.force_to_top).__module__}.{type(var_11.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.force_to_top) == 0
    assert f'{type(var_11.skip).__module__}.{type(var_11.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.skip) == 19
    assert f'{type(var_11.extend_skip).__module__}.{type(var_11.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extend_skip) == 0
    assert f'{type(var_11.skip_glob).__module__}.{type(var_11.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.skip_glob) == 0
    assert f'{type(var_11.extend_skip_glob).__module__}.{type(var_11.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extend_skip_glob) == 0
    assert var_11.skip_gitignore is False
    assert var_11.line_length == 88
    assert var_11.wrap_length == 79
    assert var_11.line_ending == ''
    assert var_11.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_11.no_sections is False
    assert f'{type(var_11.known_future_library).__module__}.{type(var_11.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_future_library) == 1
    assert f'{type(var_11.known_third_party).__module__}.{type(var_11.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_third_party) == 0
    assert f'{type(var_11.known_first_party).__module__}.{type(var_11.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_first_party) == 0
    assert f'{type(var_11.known_local_folder).__module__}.{type(var_11.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_local_folder) == 0
    assert f'{type(var_11.known_standard_library).__module__}.{type(var_11.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.known_standard_library) == 234
    assert f'{type(var_11.extra_standard_library).__module__}.{type(var_11.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.extra_standard_library) == 0
    assert var_11.known_other == {}
    assert var_11.multi_line_output == module_2.WrapModes.GRID
    assert var_11.forced_separate == ()
    assert var_11.indent == '    '
    assert var_11.comment_prefix == '  #'
    assert var_11.length_sort is False
    assert var_11.length_sort_straight is False
    assert f'{type(var_11.length_sort_sections).__module__}.{type(var_11.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.length_sort_sections) == 0
    assert f'{type(var_11.add_imports).__module__}.{type(var_11.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.add_imports) == 0
    assert f'{type(var_11.remove_imports).__module__}.{type(var_11.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.remove_imports) == 0
    assert var_11.append_only is False
    assert var_11.reverse_relative is False
    assert var_11.force_single_line is False
    assert var_11.single_line_exclusions == ()
    assert var_11.default_section == 'THIRDPARTY'
    assert var_11.import_headings == {'FUTURE': '# Future imports'}
    assert var_11.import_footers == {}
    assert var_11.balanced_wrapping is False
    assert var_11.use_parentheses is False
    assert var_11.order_by_type is True
    assert var_11.atomic is False
    assert var_11.lines_before_imports == -1
    assert var_11.lines_after_imports == -1
    assert var_11.lines_between_sections == 1
    assert var_11.lines_between_types == 0
    assert var_11.combine_as_imports is False
    assert var_11.combine_star is False
    assert var_11.include_trailing_comma is False
    assert var_11.from_first is False
    assert var_11.verbose is False
    assert var_11.quiet is False
    assert var_11.force_adds is False
    assert var_11.force_alphabetical_sort_within_sections is False
    assert var_11.force_alphabetical_sort is False
    assert var_11.force_grid_wrap == 0
    assert var_11.force_sort_within_sections is False
    assert var_11.lexicographical is False
    assert var_11.group_by_package is False
    assert var_11.ignore_whitespace is False
    assert f'{type(var_11.no_lines_before).__module__}.{type(var_11.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.no_lines_before) == 0
    assert var_11.no_inline_sort is False
    assert var_11.ignore_comments is False
    assert var_11.case_sensitive is False
    assert f'{type(var_11.sources).__module__}.{type(var_11.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_11.sources) == 2
    assert var_11.virtual_env == ''
    assert var_11.conda_env == ''
    assert var_11.ensure_newline_before_comments is False
    assert var_11.directory == '/workspace'
    assert var_11.profile == ''
    assert var_11.honor_noqa is False
    assert f'{type(var_11.src_paths).__module__}.{type(var_11.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_11.src_paths) == 2
    assert var_11.remove_redundant_aliases is False
    assert var_11.float_to_top is False
    assert var_11.filter_files is False
    assert var_11.formatter == ''
    assert var_11.formatting_function is None
    assert var_11.color_output is False
    assert f'{type(var_11.treat_comments_as_code).__module__}.{type(var_11.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.treat_comments_as_code) == 0
    assert var_11.treat_all_comments_as_code is False
    assert f'{type(var_11.supported_extensions).__module__}.{type(var_11.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.supported_extensions) == 4
    assert f'{type(var_11.blocked_extensions).__module__}.{type(var_11.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.blocked_extensions) == 1
    assert f'{type(var_11.constants).__module__}.{type(var_11.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.constants) == 0
    assert f'{type(var_11.classes).__module__}.{type(var_11.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.classes) == 0
    assert f'{type(var_11.variables).__module__}.{type(var_11.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.variables) == 0
    assert var_11.dedup_headings is False
    assert var_11.only_sections is False
    assert var_11.only_modified is False
    assert var_11.combine_straight_imports is False
    assert var_11.auto_identify_namespace_packages is True
    assert f'{type(var_11.namespace_packages).__module__}.{type(var_11.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_11.namespace_packages) == 0
    assert var_11.follow_links is True
    assert var_11.indented_import_headings is False
    assert var_11.honor_case_in_force_sorted_sections is False
    assert var_11.sort_relative_in_force_sorted_sections is False
    assert var_11.overwrite_in_place is False
    assert var_11.reverse_sort is False
    assert var_11.star_first is False
    assert var_11.git_ls_files == {}
    assert var_11.format_error == '{error}: {message}'
    assert var_11.format_success == '{success}: {message}'
    assert var_11.sort_order == 'natural'
    assert var_11.sort_reexports is False
    assert var_11.split_on_trailing_comma is False
    assert module_4.TYPE_CHECKING is False
    assert module_4.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_4.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.FIRSTPARTY == 'FIRSTPARTY'
    assert module_4.FUTURE == 'FUTURE'
    assert module_4.LOCALFOLDER == 'LOCALFOLDER'
    assert module_4.STDLIB == 'STDLIB'
    assert module_4.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_4.CYTHON_EXTENSIONS).__module__}.{type(module_4.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_4.SUPPORTED_EXTENSIONS).__module__}.{type(module_4.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_4.BLOCKED_EXTENSIONS).__module__}.{type(module_4.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.BLOCKED_EXTENSIONS) == 1
    assert module_4.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_4.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_4.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_4.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_4.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_4.DEFAULT_SKIP).__module__}.{type(module_4.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_SKIP) == 19
    assert module_4.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_4.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_4.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_4.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_4.KNOWN_PREFIX == 'known_'
    assert module_4.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_4.RUNTIME_SOURCE == 'runtime'
    assert module_4.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_4.DEFAULT_CONFIG).__module__}.{type(module_4.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_4.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_4.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_4.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip).__module__}.{type(module_4.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_4.DEFAULT_CONFIG.skip_gitignore is False
    assert module_4.DEFAULT_CONFIG.line_length == 79
    assert module_4.DEFAULT_CONFIG.wrap_length == 0
    assert module_4.DEFAULT_CONFIG.line_ending == ''
    assert module_4.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_4.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_4.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_4.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_4.DEFAULT_CONFIG.known_other == {}
    assert module_4.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_4.DEFAULT_CONFIG.forced_separate == ()
    assert module_4.DEFAULT_CONFIG.indent == '    '
    assert module_4.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_4.DEFAULT_CONFIG.length_sort is False
    assert module_4.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_4.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_4.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.add_imports).__module__}.{type(module_4.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_4.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.remove_imports) == 0
    assert module_4.DEFAULT_CONFIG.append_only is False
    assert module_4.DEFAULT_CONFIG.reverse_relative is False
    assert module_4.DEFAULT_CONFIG.force_single_line is False
    assert module_4.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_4.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_4.DEFAULT_CONFIG.import_headings == {}
    assert module_4.DEFAULT_CONFIG.import_footers == {}
    assert module_4.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_4.DEFAULT_CONFIG.use_parentheses is False
    assert module_4.DEFAULT_CONFIG.order_by_type is True
    assert module_4.DEFAULT_CONFIG.atomic is False
    assert module_4.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_4.DEFAULT_CONFIG.lines_between_types == 0
    assert module_4.DEFAULT_CONFIG.combine_as_imports is False
    assert module_4.DEFAULT_CONFIG.combine_star is False
    assert module_4.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_4.DEFAULT_CONFIG.from_first is False
    assert module_4.DEFAULT_CONFIG.verbose is False
    assert module_4.DEFAULT_CONFIG.quiet is False
    assert module_4.DEFAULT_CONFIG.force_adds is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_4.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_4.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.lexicographical is False
    assert module_4.DEFAULT_CONFIG.group_by_package is False
    assert module_4.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_4.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_4.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_4.DEFAULT_CONFIG.no_inline_sort is False
    assert module_4.DEFAULT_CONFIG.ignore_comments is False
    assert module_4.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_4.DEFAULT_CONFIG.sources).__module__}.{type(module_4.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.sources) == 1
    assert module_4.DEFAULT_CONFIG.virtual_env == ''
    assert module_4.DEFAULT_CONFIG.conda_env == ''
    assert module_4.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_4.DEFAULT_CONFIG.directory == '/workspace'
    assert module_4.DEFAULT_CONFIG.profile == ''
    assert module_4.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_4.DEFAULT_CONFIG.src_paths).__module__}.{type(module_4.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.src_paths) == 2
    assert module_4.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_4.DEFAULT_CONFIG.float_to_top is False
    assert module_4.DEFAULT_CONFIG.filter_files is False
    assert module_4.DEFAULT_CONFIG.formatter == ''
    assert module_4.DEFAULT_CONFIG.formatting_function is None
    assert module_4.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_4.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_4.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_4.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.constants).__module__}.{type(module_4.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.classes).__module__}.{type(module_4.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.variables).__module__}.{type(module_4.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.variables) == 0
    assert module_4.DEFAULT_CONFIG.dedup_headings is False
    assert module_4.DEFAULT_CONFIG.only_sections is False
    assert module_4.DEFAULT_CONFIG.only_modified is False
    assert module_4.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_4.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_4.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_4.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_4.DEFAULT_CONFIG.follow_links is True
    assert module_4.DEFAULT_CONFIG.indented_import_headings is True
    assert module_4.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_4.DEFAULT_CONFIG.reverse_sort is False
    assert module_4.DEFAULT_CONFIG.star_first is False
    assert module_4.DEFAULT_CONFIG.git_ls_files == {}
    assert module_4.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_4.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_4.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_4.DEFAULT_CONFIG.sort_reexports is False
    assert module_4.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_4.Config.known_patterns).__module__}.{type(module_4.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments).__module__}.{type(module_4.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments_end).__module__}.{type(module_4.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skips).__module__}.{type(module_4.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skip_globs).__module__}.{type(module_4.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.sorting_function).__module__}.{type(module_4.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_12 = '    '
    var_13 = module_0._indented_config(var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.settings.Config'
    assert var_13.py_version == 'py3'
    assert f'{type(var_13.force_to_top).__module__}.{type(var_13.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.force_to_top) == 0
    assert f'{type(var_13.skip).__module__}.{type(var_13.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.skip) == 19
    assert f'{type(var_13.extend_skip).__module__}.{type(var_13.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.extend_skip) == 0
    assert f'{type(var_13.skip_glob).__module__}.{type(var_13.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.skip_glob) == 0
    assert f'{type(var_13.extend_skip_glob).__module__}.{type(var_13.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.extend_skip_glob) == 0
    assert var_13.skip_gitignore is False
    assert var_13.line_length == 84
    assert var_13.wrap_length == 75
    assert var_13.line_ending == ''
    assert var_13.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_13.no_sections is False
    assert f'{type(var_13.known_future_library).__module__}.{type(var_13.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.known_future_library) == 1
    assert f'{type(var_13.known_third_party).__module__}.{type(var_13.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.known_third_party) == 0
    assert f'{type(var_13.known_first_party).__module__}.{type(var_13.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.known_first_party) == 0
    assert f'{type(var_13.known_local_folder).__module__}.{type(var_13.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.known_local_folder) == 0
    assert f'{type(var_13.known_standard_library).__module__}.{type(var_13.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.known_standard_library) == 234
    assert f'{type(var_13.extra_standard_library).__module__}.{type(var_13.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.extra_standard_library) == 0
    assert var_13.known_other == {}
    assert var_13.multi_line_output == module_2.WrapModes.GRID
    assert var_13.forced_separate == ()
    assert var_13.indent == '    '
    assert var_13.comment_prefix == '  #'
    assert var_13.length_sort is False
    assert var_13.length_sort_straight is False
    assert f'{type(var_13.length_sort_sections).__module__}.{type(var_13.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.length_sort_sections) == 0
    assert f'{type(var_13.add_imports).__module__}.{type(var_13.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.add_imports) == 0
    assert f'{type(var_13.remove_imports).__module__}.{type(var_13.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.remove_imports) == 0
    assert var_13.append_only is False
    assert var_13.reverse_relative is False
    assert var_13.force_single_line is False
    assert var_13.single_line_exclusions == ()
    assert var_13.default_section == 'THIRDPARTY'
    assert var_13.import_headings == {}
    assert var_13.import_footers == {}
    assert var_13.balanced_wrapping is False
    assert var_13.use_parentheses is False
    assert var_13.order_by_type is True
    assert var_13.atomic is False
    assert var_13.lines_before_imports == -1
    assert var_13.lines_after_imports == 1
    assert var_13.lines_between_sections == 1
    assert var_13.lines_between_types == 0
    assert var_13.combine_as_imports is False
    assert var_13.combine_star is False
    assert var_13.include_trailing_comma is False
    assert var_13.from_first is False
    assert var_13.verbose is False
    assert var_13.quiet is False
    assert var_13.force_adds is False
    assert var_13.force_alphabetical_sort_within_sections is False
    assert var_13.force_alphabetical_sort is False
    assert var_13.force_grid_wrap == 0
    assert var_13.force_sort_within_sections is False
    assert var_13.lexicographical is False
    assert var_13.group_by_package is False
    assert var_13.ignore_whitespace is False
    assert f'{type(var_13.no_lines_before).__module__}.{type(var_13.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.no_lines_before) == 0
    assert var_13.no_inline_sort is False
    assert var_13.ignore_comments is False
    assert var_13.case_sensitive is False
    assert f'{type(var_13.sources).__module__}.{type(var_13.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_13.sources) == 2
    assert var_13.virtual_env == ''
    assert var_13.conda_env == ''
    assert var_13.ensure_newline_before_comments is False
    assert var_13.directory == '/workspace'
    assert var_13.profile == ''
    assert var_13.honor_noqa is False
    assert f'{type(var_13.src_paths).__module__}.{type(var_13.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_13.src_paths) == 2
    assert var_13.remove_redundant_aliases is False
    assert var_13.float_to_top is False
    assert var_13.filter_files is False
    assert var_13.formatter == ''
    assert var_13.formatting_function is None
    assert var_13.color_output is False
    assert f'{type(var_13.treat_comments_as_code).__module__}.{type(var_13.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.treat_comments_as_code) == 0
    assert var_13.treat_all_comments_as_code is False
    assert f'{type(var_13.supported_extensions).__module__}.{type(var_13.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.supported_extensions) == 4
    assert f'{type(var_13.blocked_extensions).__module__}.{type(var_13.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.blocked_extensions) == 1
    assert f'{type(var_13.constants).__module__}.{type(var_13.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.constants) == 0
    assert f'{type(var_13.classes).__module__}.{type(var_13.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.classes) == 0
    assert f'{type(var_13.variables).__module__}.{type(var_13.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.variables) == 0
    assert var_13.dedup_headings is False
    assert var_13.only_sections is False
    assert var_13.only_modified is False
    assert var_13.combine_straight_imports is False
    assert var_13.auto_identify_namespace_packages is True
    assert f'{type(var_13.namespace_packages).__module__}.{type(var_13.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_13.namespace_packages) == 0
    assert var_13.follow_links is True
    assert var_13.indented_import_headings is False
    assert var_13.honor_case_in_force_sorted_sections is False
    assert var_13.sort_relative_in_force_sorted_sections is False
    assert var_13.overwrite_in_place is False
    assert var_13.reverse_sort is False
    assert var_13.star_first is False
    assert var_13.git_ls_files == {}
    assert var_13.format_error == '{error}: {message}'
    assert var_13.format_success == '{success}: {message}'
    assert var_13.sort_order == 'natural'
    assert var_13.sort_reexports is False
    assert var_13.split_on_trailing_comma is False
    assert f'{type(module_0.DEFAULT_CONFIG).__module__}.{type(module_0.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_0.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_0.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_0.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip).__module__}.{type(module_0.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_0.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_0.DEFAULT_CONFIG.skip_gitignore is False
    assert module_0.DEFAULT_CONFIG.line_length == 79
    assert module_0.DEFAULT_CONFIG.wrap_length == 0
    assert module_0.DEFAULT_CONFIG.line_ending == ''
    assert module_0.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_0.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_0.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_0.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_0.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_0.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_0.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_0.DEFAULT_CONFIG.known_other == {}
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_0.DEFAULT_CONFIG.forced_separate == ()
    assert module_0.DEFAULT_CONFIG.indent == '    '
    assert module_0.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_0.DEFAULT_CONFIG.length_sort is False
    assert module_0.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_0.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_0.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.add_imports).__module__}.{type(module_0.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_0.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.remove_imports) == 0
    assert module_0.DEFAULT_CONFIG.append_only is False
    assert module_0.DEFAULT_CONFIG.reverse_relative is False
    assert module_0.DEFAULT_CONFIG.force_single_line is False
    assert module_0.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_0.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_0.DEFAULT_CONFIG.import_headings == {}
    assert module_0.DEFAULT_CONFIG.import_footers == {}
    assert module_0.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_0.DEFAULT_CONFIG.use_parentheses is False
    assert module_0.DEFAULT_CONFIG.order_by_type is True
    assert module_0.DEFAULT_CONFIG.atomic is False
    assert module_0.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_0.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_0.DEFAULT_CONFIG.lines_between_types == 0
    assert module_0.DEFAULT_CONFIG.combine_as_imports is False
    assert module_0.DEFAULT_CONFIG.combine_star is False
    assert module_0.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_0.DEFAULT_CONFIG.from_first is False
    assert module_0.DEFAULT_CONFIG.verbose is False
    assert module_0.DEFAULT_CONFIG.quiet is False
    assert module_0.DEFAULT_CONFIG.force_adds is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_0.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_0.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_0.DEFAULT_CONFIG.lexicographical is False
    assert module_0.DEFAULT_CONFIG.group_by_package is False
    assert module_0.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_0.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_0.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_0.DEFAULT_CONFIG.no_inline_sort is False
    assert module_0.DEFAULT_CONFIG.ignore_comments is False
    assert module_0.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_0.DEFAULT_CONFIG.sources).__module__}.{type(module_0.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.sources) == 1
    assert module_0.DEFAULT_CONFIG.virtual_env == ''
    assert module_0.DEFAULT_CONFIG.conda_env == ''
    assert module_0.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_0.DEFAULT_CONFIG.directory == '/workspace'
    assert module_0.DEFAULT_CONFIG.profile == ''
    assert module_0.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_0.DEFAULT_CONFIG.src_paths).__module__}.{type(module_0.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_0.DEFAULT_CONFIG.src_paths) == 2
    assert module_0.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_0.DEFAULT_CONFIG.float_to_top is False
    assert module_0.DEFAULT_CONFIG.filter_files is False
    assert module_0.DEFAULT_CONFIG.formatter == ''
    assert module_0.DEFAULT_CONFIG.formatting_function is None
    assert module_0.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_0.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_0.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_0.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_0.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_0.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_0.DEFAULT_CONFIG.constants).__module__}.{type(module_0.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.classes).__module__}.{type(module_0.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_0.DEFAULT_CONFIG.variables).__module__}.{type(module_0.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.variables) == 0
    assert module_0.DEFAULT_CONFIG.dedup_headings is False
    assert module_0.DEFAULT_CONFIG.only_sections is False
    assert module_0.DEFAULT_CONFIG.only_modified is False
    assert module_0.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_0.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_0.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_0.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_0.DEFAULT_CONFIG.follow_links is True
    assert module_0.DEFAULT_CONFIG.indented_import_headings is True
    assert module_0.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_0.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_0.DEFAULT_CONFIG.reverse_sort is False
    assert module_0.DEFAULT_CONFIG.star_first is False
    assert module_0.DEFAULT_CONFIG.git_ls_files == {}
    assert module_0.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_0.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_0.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_0.DEFAULT_CONFIG.sort_reexports is False
    assert module_0.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}
    var_14 = var_13.import_headings
    var_15 = bool(var_13.import_headings == {})
    assert var_15 is True

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = "8'\\CH]X<"
    var_1 = [var_0]
    module_0.process(var_1, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'from syimport aI!\nfoos ort aF\\'
    var_1 = [var_0, var_0]
    module_0.process(var_1, var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'import sys'
    var_1 = [var_0]
    var_2 = 'add_imports'
    var_3 = {var_2: var_1}
    var_4 = module_4.Config(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.settings.Config'
    assert var_4.py_version == 'py3'
    assert f'{type(var_4.force_to_top).__module__}.{type(var_4.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.force_to_top) == 0
    assert f'{type(var_4.skip).__module__}.{type(var_4.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip) == 19
    assert f'{type(var_4.extend_skip).__module__}.{type(var_4.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extend_skip) == 0
    assert f'{type(var_4.skip_glob).__module__}.{type(var_4.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.skip_glob) == 0
    assert f'{type(var_4.extend_skip_glob).__module__}.{type(var_4.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extend_skip_glob) == 0
    assert var_4.skip_gitignore is False
    assert var_4.line_length == 79
    assert var_4.wrap_length == 0
    assert var_4.line_ending == ''
    assert var_4.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_4.no_sections is False
    assert f'{type(var_4.known_future_library).__module__}.{type(var_4.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_future_library) == 1
    assert f'{type(var_4.known_third_party).__module__}.{type(var_4.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_third_party) == 0
    assert f'{type(var_4.known_first_party).__module__}.{type(var_4.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_first_party) == 0
    assert f'{type(var_4.known_local_folder).__module__}.{type(var_4.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_local_folder) == 0
    assert f'{type(var_4.known_standard_library).__module__}.{type(var_4.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.known_standard_library) == 234
    assert f'{type(var_4.extra_standard_library).__module__}.{type(var_4.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.extra_standard_library) == 0
    assert var_4.known_other == {}
    assert var_4.multi_line_output == module_2.WrapModes.GRID
    assert var_4.forced_separate == ()
    assert var_4.indent == '    '
    assert var_4.comment_prefix == '  #'
    assert var_4.length_sort is False
    assert var_4.length_sort_straight is False
    assert f'{type(var_4.length_sort_sections).__module__}.{type(var_4.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.length_sort_sections) == 0
    assert f'{type(var_4.add_imports).__module__}.{type(var_4.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.add_imports) == 1
    assert f'{type(var_4.remove_imports).__module__}.{type(var_4.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.remove_imports) == 0
    assert var_4.append_only is False
    assert var_4.reverse_relative is False
    assert var_4.force_single_line is False
    assert var_4.single_line_exclusions == ()
    assert var_4.default_section == 'THIRDPARTY'
    assert var_4.import_headings == {}
    assert var_4.import_footers == {}
    assert var_4.balanced_wrapping is False
    assert var_4.use_parentheses is False
    assert var_4.order_by_type is True
    assert var_4.atomic is False
    assert var_4.lines_before_imports == -1
    assert var_4.lines_after_imports == -1
    assert var_4.lines_between_sections == 1
    assert var_4.lines_between_types == 0
    assert var_4.combine_as_imports is False
    assert var_4.combine_star is False
    assert var_4.include_trailing_comma is False
    assert var_4.from_first is False
    assert var_4.verbose is False
    assert var_4.quiet is False
    assert var_4.force_adds is False
    assert var_4.force_alphabetical_sort_within_sections is False
    assert var_4.force_alphabetical_sort is False
    assert var_4.force_grid_wrap == 0
    assert var_4.force_sort_within_sections is False
    assert var_4.lexicographical is False
    assert var_4.group_by_package is False
    assert var_4.ignore_whitespace is False
    assert f'{type(var_4.no_lines_before).__module__}.{type(var_4.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.no_lines_before) == 0
    assert var_4.no_inline_sort is False
    assert var_4.ignore_comments is False
    assert var_4.case_sensitive is False
    assert f'{type(var_4.sources).__module__}.{type(var_4.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_4.sources) == 2
    assert var_4.virtual_env == ''
    assert var_4.conda_env == ''
    assert var_4.ensure_newline_before_comments is False
    assert var_4.directory == '/workspace'
    assert var_4.profile == ''
    assert var_4.honor_noqa is False
    assert f'{type(var_4.src_paths).__module__}.{type(var_4.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_4.src_paths) == 2
    assert var_4.remove_redundant_aliases is False
    assert var_4.float_to_top is False
    assert var_4.filter_files is False
    assert var_4.formatter == ''
    assert var_4.formatting_function is None
    assert var_4.color_output is False
    assert f'{type(var_4.treat_comments_as_code).__module__}.{type(var_4.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.treat_comments_as_code) == 0
    assert var_4.treat_all_comments_as_code is False
    assert f'{type(var_4.supported_extensions).__module__}.{type(var_4.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.supported_extensions) == 4
    assert f'{type(var_4.blocked_extensions).__module__}.{type(var_4.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.blocked_extensions) == 1
    assert f'{type(var_4.constants).__module__}.{type(var_4.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.constants) == 0
    assert f'{type(var_4.classes).__module__}.{type(var_4.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.classes) == 0
    assert f'{type(var_4.variables).__module__}.{type(var_4.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.variables) == 0
    assert var_4.dedup_headings is False
    assert var_4.only_sections is False
    assert var_4.only_modified is False
    assert var_4.combine_straight_imports is False
    assert var_4.auto_identify_namespace_packages is True
    assert f'{type(var_4.namespace_packages).__module__}.{type(var_4.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.namespace_packages) == 0
    assert var_4.follow_links is True
    assert var_4.indented_import_headings is True
    assert var_4.honor_case_in_force_sorted_sections is False
    assert var_4.sort_relative_in_force_sorted_sections is False
    assert var_4.overwrite_in_place is False
    assert var_4.reverse_sort is False
    assert var_4.star_first is False
    assert var_4.git_ls_files == {}
    assert var_4.format_error == '{error}: {message}'
    assert var_4.format_success == '{success}: {message}'
    assert var_4.sort_order == 'natural'
    assert var_4.sort_reexports is False
    assert var_4.split_on_trailing_comma is False
    assert module_4.TYPE_CHECKING is False
    assert module_4.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_4.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.FIRSTPARTY == 'FIRSTPARTY'
    assert module_4.FUTURE == 'FUTURE'
    assert module_4.LOCALFOLDER == 'LOCALFOLDER'
    assert module_4.STDLIB == 'STDLIB'
    assert module_4.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_4.CYTHON_EXTENSIONS).__module__}.{type(module_4.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_4.SUPPORTED_EXTENSIONS).__module__}.{type(module_4.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_4.BLOCKED_EXTENSIONS).__module__}.{type(module_4.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.BLOCKED_EXTENSIONS) == 1
    assert module_4.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_4.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_4.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_4.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_4.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_4.DEFAULT_SKIP).__module__}.{type(module_4.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_SKIP) == 19
    assert module_4.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_4.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_4.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_4.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_4.KNOWN_PREFIX == 'known_'
    assert module_4.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_4.RUNTIME_SOURCE == 'runtime'
    assert module_4.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_4.DEFAULT_CONFIG).__module__}.{type(module_4.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_4.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_4.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_4.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip).__module__}.{type(module_4.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_4.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_4.DEFAULT_CONFIG.skip_gitignore is False
    assert module_4.DEFAULT_CONFIG.line_length == 79
    assert module_4.DEFAULT_CONFIG.wrap_length == 0
    assert module_4.DEFAULT_CONFIG.line_ending == ''
    assert module_4.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_4.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_4.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_4.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_4.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_4.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_4.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_4.DEFAULT_CONFIG.known_other == {}
    assert module_4.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
    assert module_4.DEFAULT_CONFIG.forced_separate == ()
    assert module_4.DEFAULT_CONFIG.indent == '    '
    assert module_4.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_4.DEFAULT_CONFIG.length_sort is False
    assert module_4.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_4.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_4.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.add_imports).__module__}.{type(module_4.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_4.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.remove_imports) == 0
    assert module_4.DEFAULT_CONFIG.append_only is False
    assert module_4.DEFAULT_CONFIG.reverse_relative is False
    assert module_4.DEFAULT_CONFIG.force_single_line is False
    assert module_4.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_4.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_4.DEFAULT_CONFIG.import_headings == {}
    assert module_4.DEFAULT_CONFIG.import_footers == {}
    assert module_4.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_4.DEFAULT_CONFIG.use_parentheses is False
    assert module_4.DEFAULT_CONFIG.order_by_type is True
    assert module_4.DEFAULT_CONFIG.atomic is False
    assert module_4.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_4.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_4.DEFAULT_CONFIG.lines_between_types == 0
    assert module_4.DEFAULT_CONFIG.combine_as_imports is False
    assert module_4.DEFAULT_CONFIG.combine_star is False
    assert module_4.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_4.DEFAULT_CONFIG.from_first is False
    assert module_4.DEFAULT_CONFIG.verbose is False
    assert module_4.DEFAULT_CONFIG.quiet is False
    assert module_4.DEFAULT_CONFIG.force_adds is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_4.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_4.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_4.DEFAULT_CONFIG.lexicographical is False
    assert module_4.DEFAULT_CONFIG.group_by_package is False
    assert module_4.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_4.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_4.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_4.DEFAULT_CONFIG.no_inline_sort is False
    assert module_4.DEFAULT_CONFIG.ignore_comments is False
    assert module_4.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_4.DEFAULT_CONFIG.sources).__module__}.{type(module_4.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.sources) == 1
    assert module_4.DEFAULT_CONFIG.virtual_env == ''
    assert module_4.DEFAULT_CONFIG.conda_env == ''
    assert module_4.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_4.DEFAULT_CONFIG.directory == '/workspace'
    assert module_4.DEFAULT_CONFIG.profile == ''
    assert module_4.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_4.DEFAULT_CONFIG.src_paths).__module__}.{type(module_4.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_4.DEFAULT_CONFIG.src_paths) == 2
    assert module_4.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_4.DEFAULT_CONFIG.float_to_top is False
    assert module_4.DEFAULT_CONFIG.filter_files is False
    assert module_4.DEFAULT_CONFIG.formatter == ''
    assert module_4.DEFAULT_CONFIG.formatting_function is None
    assert module_4.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_4.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_4.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_4.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_4.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_4.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_4.DEFAULT_CONFIG.constants).__module__}.{type(module_4.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.classes).__module__}.{type(module_4.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_4.DEFAULT_CONFIG.variables).__module__}.{type(module_4.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.variables) == 0
    assert module_4.DEFAULT_CONFIG.dedup_headings is False
    assert module_4.DEFAULT_CONFIG.only_sections is False
    assert module_4.DEFAULT_CONFIG.only_modified is False
    assert module_4.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_4.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_4.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_4.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_4.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_4.DEFAULT_CONFIG.follow_links is True
    assert module_4.DEFAULT_CONFIG.indented_import_headings is True
    assert module_4.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_4.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_4.DEFAULT_CONFIG.reverse_sort is False
    assert module_4.DEFAULT_CONFIG.star_first is False
    assert module_4.DEFAULT_CONFIG.git_ls_files == {}
    assert module_4.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_4.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_4.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_4.DEFAULT_CONFIG.sort_reexports is False
    assert module_4.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_4.Config.known_patterns).__module__}.{type(module_4.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments).__module__}.{type(module_4.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.section_comments_end).__module__}.{type(module_4.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skips).__module__}.{type(module_4.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.skip_globs).__module__}.{type(module_4.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Config.sorting_function).__module__}.{type(module_4.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_5 = 'import sys'
    var_6 = None
    module_0.process(var_6, var_5, var_6, config=var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = [var_0]
    module_0.process(var_1, var_1)