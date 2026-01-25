# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_0
import re as module_2

import isort.wrap as module_1
import isort.wrap_modes as module_3
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    module_1.import_statement(var_0, var_0, line_separator=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '\\Cf=05.p9M|!"),8mV?'
    var_1 = module_2.escape(var_0)
    assert var_1 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_2 = '|C#o,\x0cOC`r'
    var_3 = None
    var_4 = False
    var_5 = module_1.import_statement(var_2, var_3, explode=var_4)
    assert var_5 == ''
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_6 = [var_0, var_0, var_5, var_2]
    var_7 = module_1.import_statement(var_1, var_6, line_separator=var_1)
    assert var_7 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.p9M|!"),8mV?, \\Cf=05.p9M|!"),8mV?, , |C#o,\x0cOC`r)'
    var_8 = module_1.line(var_7, var_7)
    assert var_8 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.p9M|!"),8mV?, \\Cf=05.\\\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.p9M|!"),8mV?, \\Cf=05.p9M|!"),8mV?, , |C#o,\x0cOC`r)    p9M|!"),8mV?, , |C  #o,\x0cOC`r)'
    module_1.import_statement(var_1, var_1, config=var_1, explode=var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.import_statement(var_0, var_0, multi_line_output=var_0)

def test_case_3():
    var_0 = '\\C3fq09M|\tB"),8mV?'
    var_1 = module_2.purge()
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_2 = module_1.import_statement(var_0, var_1, line_separator=var_0)
    assert var_2 == ''
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False

def test_case_4():
    var_0 = '\\C3fq09M|\tB"),8mV?'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_1.import_statement(var_0, var_1, line_separator=var_0)
    assert var_2 == '\\C3fq09M|\tB"),8mV?(\\C3fq09M|\tB"),8mV?, \\C3fq09M|\tB"),8mV?, \\C3fq09M|\tB"),8mV?, \\C3fq09M|\tB"),8mV?)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False

def test_case_5():
    var_0 = ''
    var_1 = module_1.line(var_0, var_0)
    assert var_1 == ''
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\\Cf=05.p9M|!"),8mV?'
    var_1 = module_2.escape(var_0)
    assert var_1 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_2 = '|C#o,\x0c~O(>`r'
    var_3 = [var_1, var_2, var_2]
    var_4 = None
    var_5 = [var_0, var_2, var_1]
    var_6 = module_1.import_statement(var_1, var_5, line_separator=var_1)
    assert var_6 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.p9M|!"),8mV?, |C#o,\x0c~O(>`r, \\\\Cf=05\\.p9M\\|!"\\),8mV\\?)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_7 = module_1.line(var_1, var_1)
    assert var_7 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?'
    var_8 = module_1.line(var_6, var_6)
    assert var_8 == '\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.\\\\\\Cf=05\\.p9M\\|!"\\),8mV\\?(\\Cf=05.p9M|!"),8mV?, |C#o,\x0c~O(>`r, \\\\Cf=05\\.p9M\\|!"\\),8mV\\?)    p9M|!"),8mV?, |C  #o,\x0c~O(>`r, \\\\Cf=05\\.p9M\\|!"\\),8mV\\?)'
    var_9 = '<'
    var_10 = module_1.line(var_3, var_9)
    module_1.import_statement(var_4, var_1, config=var_1, multi_line_output=var_1, explode=var_4)

def test_case_7():
    var_0 = '\\C3fq09M|\tB")8mV?'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_1.import_statement(var_0, var_1, line_separator=var_0)
    assert var_2 == '\\C3fq09M|\tB")8mV?(\\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_3 = module_1.line(var_2, var_0)
    assert var_3 == '\\C3fq09M|\tB")8mV?(\\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?, \\C3fq09M|\tB")8mV?)'

def test_case_8():
    var_0 = '\\f=05.p9M|!"),8m?'
    var_1 = module_2.escape(var_0)
    assert var_1 == '\\\\f=05\\.p9M\\|!"\\),8m\\?'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_2 = '|C#,\x0cOC`r'
    var_3 = [var_0, var_0, var_2, var_2]
    var_4 = module_1.import_statement(var_1, var_3, line_separator=var_1)
    assert var_4 == '\\\\f=05\\.p9M\\|!"\\),8m\\?(\\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, |C#,\x0cOC`r, |C#,\x0cOC`r)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_5 = module_1.line(var_4, var_4)
    assert var_5 == '\\\\f=05\\.p9M\\|!"\\),8m\\?(\\f=05.p9M|!"),8m?, \\f=05.\\\\\\f=05\\.p9M\\|!"\\),8m\\?(\\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, |C#,\x0cOC`r, |C#,\x0cOC`r)    p9M|!"),8m?, |C  #,\x0cOC`r, |C#,\x0cOC`r)'

def test_case_9():
    var_0 = '\\f=05.p9M|!"),8m?'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_1.import_statement(var_0, var_1, line_separator=var_0)
    assert var_2 == '\\f=05.p9M|!"),8m?(\\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_3 = module_1.line(var_2, var_2)
    assert var_3 == '\\f=05.p9M|!"),8m?(\\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, \\f=05.\\\\f=05.p9M|!"),8m?(\\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?, \\f=05.p9M|!"),8m?)    p9M|!"),8m?).p9M|!"),8m?, \\f=05'

def test_case_10():
    var_0 = 'gGo@K!W!zl:$HE'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = None
    var_3 = module_1.import_statement(var_0, var_1, multi_line_output=var_2)
    assert var_3 == 'gGo@K!W!zl:$HE(gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE)'
    assert f'{type(module_1.DEFAULT_CONFIG).__module__}.{type(module_1.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_1.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_1.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_1.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip).__module__}.{type(module_1.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_1.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_1.DEFAULT_CONFIG.skip_gitignore is False
    assert module_1.DEFAULT_CONFIG.line_length == 79
    assert module_1.DEFAULT_CONFIG.wrap_length == 0
    assert module_1.DEFAULT_CONFIG.line_ending == ''
    assert module_1.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_1.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_1.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_1.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_1.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_1.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_1.DEFAULT_CONFIG.known_other == {}
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_1.DEFAULT_CONFIG.forced_separate == ()
    assert module_1.DEFAULT_CONFIG.indent == '    '
    assert module_1.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_1.DEFAULT_CONFIG.length_sort is False
    assert module_1.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_1.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_1.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.add_imports).__module__}.{type(module_1.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_1.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.remove_imports) == 0
    assert module_1.DEFAULT_CONFIG.append_only is False
    assert module_1.DEFAULT_CONFIG.reverse_relative is False
    assert module_1.DEFAULT_CONFIG.force_single_line is False
    assert module_1.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_1.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_1.DEFAULT_CONFIG.import_headings == {}
    assert module_1.DEFAULT_CONFIG.import_footers == {}
    assert module_1.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_1.DEFAULT_CONFIG.use_parentheses is False
    assert module_1.DEFAULT_CONFIG.order_by_type is True
    assert module_1.DEFAULT_CONFIG.atomic is False
    assert module_1.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_1.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_1.DEFAULT_CONFIG.lines_between_types == 0
    assert module_1.DEFAULT_CONFIG.combine_as_imports is False
    assert module_1.DEFAULT_CONFIG.combine_star is False
    assert module_1.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_1.DEFAULT_CONFIG.from_first is False
    assert module_1.DEFAULT_CONFIG.verbose is False
    assert module_1.DEFAULT_CONFIG.quiet is False
    assert module_1.DEFAULT_CONFIG.force_adds is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_1.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_1.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_1.DEFAULT_CONFIG.lexicographical is False
    assert module_1.DEFAULT_CONFIG.group_by_package is False
    assert module_1.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_1.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_1.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_1.DEFAULT_CONFIG.no_inline_sort is False
    assert module_1.DEFAULT_CONFIG.ignore_comments is False
    assert module_1.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_1.DEFAULT_CONFIG.sources).__module__}.{type(module_1.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.sources) == 1
    assert module_1.DEFAULT_CONFIG.virtual_env == ''
    assert module_1.DEFAULT_CONFIG.conda_env == ''
    assert module_1.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_1.DEFAULT_CONFIG.directory == '/workspace'
    assert module_1.DEFAULT_CONFIG.profile == ''
    assert module_1.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_1.DEFAULT_CONFIG.src_paths).__module__}.{type(module_1.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_1.DEFAULT_CONFIG.src_paths) == 2
    assert module_1.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_1.DEFAULT_CONFIG.float_to_top is False
    assert module_1.DEFAULT_CONFIG.filter_files is False
    assert module_1.DEFAULT_CONFIG.formatter == ''
    assert module_1.DEFAULT_CONFIG.formatting_function is None
    assert module_1.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_1.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_1.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_1.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_1.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_1.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_1.DEFAULT_CONFIG.constants).__module__}.{type(module_1.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.classes).__module__}.{type(module_1.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_1.DEFAULT_CONFIG.variables).__module__}.{type(module_1.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.variables) == 0
    assert module_1.DEFAULT_CONFIG.dedup_headings is False
    assert module_1.DEFAULT_CONFIG.only_sections is False
    assert module_1.DEFAULT_CONFIG.only_modified is False
    assert module_1.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_1.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_1.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_1.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_1.DEFAULT_CONFIG.follow_links is True
    assert module_1.DEFAULT_CONFIG.indented_import_headings is True
    assert module_1.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_1.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_1.DEFAULT_CONFIG.reverse_sort is False
    assert module_1.DEFAULT_CONFIG.star_first is False
    assert module_1.DEFAULT_CONFIG.git_ls_files == {}
    assert module_1.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_1.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_1.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_1.DEFAULT_CONFIG.sort_reexports is False
    assert module_1.DEFAULT_CONFIG.split_on_trailing_comma is False
    var_4 = 'E.l;^F&dM/\nJN[^'
    var_5 = [var_4, var_4, var_4]
    var_6 = module_1.import_statement(var_3, var_5, line_separator=var_3)
    assert var_6 == 'gGo@K!W!zl:$HE(gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE)(E.l;^F&dM/\nJN[^, E.l;^F&dM/\nJN[^, E.l;^F&dM/\nJN[^)'
    var_7 = var_5.__repr__()
    assert var_7 == "['E.l;^F&dM/\\nJN[^', 'E.l;^F&dM/\\nJN[^', 'E.l;^F&dM/\\nJN[^']"
    var_8 = module_2.finditer(var_7, var_7)
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    var_9 = module_1.line(var_6, var_2)
    assert var_9 == 'gGo@K!W!zl:$HE(gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE, gGo@K!W!zl:$HE)(E.\\None    l;^F&dM/\nJN[^).l;^F&dM/\nJN[^, E.l;^F&dM/\nJN[^, E'