# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.api as module_0
import isort.wrap_modes as module_1
import isort.settings as module_2
import tokenize as module_3
import inspect as module_4
import urllib.request as module_5
import _io as module_6
import textwrap as module_7
import stringprep as module_8
import isort.exceptions as module_9
import email._header_value_parser as module_10

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.check_stream(var_0, var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.check_code_string(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.check_file(var_0, config=var_0, file_path=var_0, extension=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '7z'
    var_1 = None
    module_0.sort_file(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'bd#@S\t\\"XMng~%>d<'
    var_1 = None
    var_2 = {}
    module_0.sort_code_string(var_0, file_path=var_0, show_diff=var_1, **var_2)

def test_case_5():
    pass

def test_case_6():
    var_0 = None
    var_1 = module_0.sort_code_string(var_0)
    assert var_1 == ''
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

def test_case_7():
    var_0 = '.RzDo5-%`H;f'
    var_1 = module_0.check_code_string(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = module_0.check_code_string(var_0, disregard_skip=var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = {}
    var_1 = []
    var_2 = 'y'
    var_3 = module_2.Config(**var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.settings.Config'
    assert var_3.py_version == 'py3'
    assert f'{type(var_3.force_to_top).__module__}.{type(var_3.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.force_to_top) == 0
    assert f'{type(var_3.skip).__module__}.{type(var_3.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.skip) == 19
    assert f'{type(var_3.extend_skip).__module__}.{type(var_3.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extend_skip) == 0
    assert f'{type(var_3.skip_glob).__module__}.{type(var_3.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.skip_glob) == 0
    assert f'{type(var_3.extend_skip_glob).__module__}.{type(var_3.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extend_skip_glob) == 0
    assert var_3.skip_gitignore is False
    assert var_3.line_length == 79
    assert var_3.wrap_length == 0
    assert var_3.line_ending == ''
    assert var_3.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_3.no_sections is False
    assert f'{type(var_3.known_future_library).__module__}.{type(var_3.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_future_library) == 1
    assert f'{type(var_3.known_third_party).__module__}.{type(var_3.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_third_party) == 0
    assert f'{type(var_3.known_first_party).__module__}.{type(var_3.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_first_party) == 0
    assert f'{type(var_3.known_local_folder).__module__}.{type(var_3.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_local_folder) == 0
    assert f'{type(var_3.known_standard_library).__module__}.{type(var_3.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.known_standard_library) == 234
    assert f'{type(var_3.extra_standard_library).__module__}.{type(var_3.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.extra_standard_library) == 0
    assert var_3.known_other == {}
    assert var_3.multi_line_output == module_1.WrapModes.GRID
    assert var_3.forced_separate == ()
    assert var_3.indent == '    '
    assert var_3.comment_prefix == '  #'
    assert var_3.length_sort is False
    assert var_3.length_sort_straight is False
    assert f'{type(var_3.length_sort_sections).__module__}.{type(var_3.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.length_sort_sections) == 0
    assert f'{type(var_3.add_imports).__module__}.{type(var_3.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.add_imports) == 0
    assert f'{type(var_3.remove_imports).__module__}.{type(var_3.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.remove_imports) == 0
    assert var_3.append_only is False
    assert var_3.reverse_relative is False
    assert var_3.force_single_line is False
    assert var_3.single_line_exclusions == ()
    assert var_3.default_section == 'THIRDPARTY'
    assert var_3.import_headings == {}
    assert var_3.import_footers == {}
    assert var_3.balanced_wrapping is False
    assert var_3.use_parentheses is False
    assert var_3.order_by_type is True
    assert var_3.atomic is False
    assert var_3.lines_before_imports == -1
    assert var_3.lines_after_imports == -1
    assert var_3.lines_between_sections == 1
    assert var_3.lines_between_types == 0
    assert var_3.combine_as_imports is False
    assert var_3.combine_star is False
    assert var_3.include_trailing_comma is False
    assert var_3.from_first is False
    assert var_3.verbose is False
    assert var_3.quiet is False
    assert var_3.force_adds is False
    assert var_3.force_alphabetical_sort_within_sections is False
    assert var_3.force_alphabetical_sort is False
    assert var_3.force_grid_wrap == 0
    assert var_3.force_sort_within_sections is False
    assert var_3.lexicographical is False
    assert var_3.group_by_package is False
    assert var_3.ignore_whitespace is False
    assert f'{type(var_3.no_lines_before).__module__}.{type(var_3.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.no_lines_before) == 0
    assert var_3.no_inline_sort is False
    assert var_3.ignore_comments is False
    assert var_3.case_sensitive is False
    assert f'{type(var_3.sources).__module__}.{type(var_3.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_3.sources) == 1
    assert var_3.virtual_env == ''
    assert var_3.conda_env == ''
    assert var_3.ensure_newline_before_comments is False
    assert var_3.directory == '/workspace/run'
    assert var_3.profile == ''
    assert var_3.honor_noqa is False
    assert f'{type(var_3.src_paths).__module__}.{type(var_3.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_3.src_paths) == 2
    assert var_3.remove_redundant_aliases is False
    assert var_3.float_to_top is False
    assert var_3.filter_files is False
    assert var_3.formatter == ''
    assert var_3.formatting_function is None
    assert var_3.color_output is False
    assert f'{type(var_3.treat_comments_as_code).__module__}.{type(var_3.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.treat_comments_as_code) == 0
    assert var_3.treat_all_comments_as_code is False
    assert f'{type(var_3.supported_extensions).__module__}.{type(var_3.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.supported_extensions) == 4
    assert f'{type(var_3.blocked_extensions).__module__}.{type(var_3.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.blocked_extensions) == 1
    assert f'{type(var_3.constants).__module__}.{type(var_3.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.constants) == 0
    assert f'{type(var_3.classes).__module__}.{type(var_3.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.classes) == 0
    assert f'{type(var_3.variables).__module__}.{type(var_3.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.variables) == 0
    assert var_3.dedup_headings is False
    assert var_3.only_sections is False
    assert var_3.only_modified is False
    assert var_3.combine_straight_imports is False
    assert var_3.auto_identify_namespace_packages is True
    assert f'{type(var_3.namespace_packages).__module__}.{type(var_3.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_3.namespace_packages) == 0
    assert var_3.follow_links is True
    assert var_3.indented_import_headings is True
    assert var_3.honor_case_in_force_sorted_sections is False
    assert var_3.sort_relative_in_force_sorted_sections is False
    assert var_3.overwrite_in_place is False
    assert var_3.reverse_sort is False
    assert var_3.star_first is False
    assert var_3.git_ls_files == {}
    assert var_3.format_error == '{error}: {message}'
    assert var_3.format_success == '{success}: {message}'
    assert var_3.sort_order == 'natural'
    assert var_3.sort_reexports is False
    assert var_3.split_on_trailing_comma is False
    assert module_2.TYPE_CHECKING is False
    assert module_2.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_2.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.FIRSTPARTY == 'FIRSTPARTY'
    assert module_2.FUTURE == 'FUTURE'
    assert module_2.LOCALFOLDER == 'LOCALFOLDER'
    assert module_2.STDLIB == 'STDLIB'
    assert module_2.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_2.CYTHON_EXTENSIONS).__module__}.{type(module_2.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_2.SUPPORTED_EXTENSIONS).__module__}.{type(module_2.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_2.BLOCKED_EXTENSIONS).__module__}.{type(module_2.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.BLOCKED_EXTENSIONS) == 1
    assert module_2.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_2.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_2.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_2.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_2.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_2.DEFAULT_SKIP).__module__}.{type(module_2.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_SKIP) == 19
    assert module_2.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_2.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_2.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_2.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_2.KNOWN_PREFIX == 'known_'
    assert module_2.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_2.RUNTIME_SOURCE == 'runtime'
    assert module_2.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_2.DEFAULT_CONFIG).__module__}.{type(module_2.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_2.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_2.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_2.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip).__module__}.{type(module_2.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_2.DEFAULT_CONFIG.skip_gitignore is False
    assert module_2.DEFAULT_CONFIG.line_length == 79
    assert module_2.DEFAULT_CONFIG.wrap_length == 0
    assert module_2.DEFAULT_CONFIG.line_ending == ''
    assert module_2.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_2.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_2.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_2.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_2.DEFAULT_CONFIG.known_other == {}
    assert module_2.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_2.DEFAULT_CONFIG.forced_separate == ()
    assert module_2.DEFAULT_CONFIG.indent == '    '
    assert module_2.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_2.DEFAULT_CONFIG.length_sort is False
    assert module_2.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_2.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_2.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.add_imports).__module__}.{type(module_2.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_2.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.remove_imports) == 0
    assert module_2.DEFAULT_CONFIG.append_only is False
    assert module_2.DEFAULT_CONFIG.reverse_relative is False
    assert module_2.DEFAULT_CONFIG.force_single_line is False
    assert module_2.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_2.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_2.DEFAULT_CONFIG.import_headings == {}
    assert module_2.DEFAULT_CONFIG.import_footers == {}
    assert module_2.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_2.DEFAULT_CONFIG.use_parentheses is False
    assert module_2.DEFAULT_CONFIG.order_by_type is True
    assert module_2.DEFAULT_CONFIG.atomic is False
    assert module_2.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_2.DEFAULT_CONFIG.lines_between_types == 0
    assert module_2.DEFAULT_CONFIG.combine_as_imports is False
    assert module_2.DEFAULT_CONFIG.combine_star is False
    assert module_2.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_2.DEFAULT_CONFIG.from_first is False
    assert module_2.DEFAULT_CONFIG.verbose is False
    assert module_2.DEFAULT_CONFIG.quiet is False
    assert module_2.DEFAULT_CONFIG.force_adds is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_2.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_2.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.lexicographical is False
    assert module_2.DEFAULT_CONFIG.group_by_package is False
    assert module_2.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_2.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_2.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_2.DEFAULT_CONFIG.no_inline_sort is False
    assert module_2.DEFAULT_CONFIG.ignore_comments is False
    assert module_2.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_2.DEFAULT_CONFIG.sources).__module__}.{type(module_2.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.sources) == 1
    assert module_2.DEFAULT_CONFIG.virtual_env == ''
    assert module_2.DEFAULT_CONFIG.conda_env == ''
    assert module_2.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_2.DEFAULT_CONFIG.directory == '/workspace/run'
    assert module_2.DEFAULT_CONFIG.profile == ''
    assert module_2.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_2.DEFAULT_CONFIG.src_paths).__module__}.{type(module_2.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.src_paths) == 2
    assert module_2.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_2.DEFAULT_CONFIG.float_to_top is False
    assert module_2.DEFAULT_CONFIG.filter_files is False
    assert module_2.DEFAULT_CONFIG.formatter == ''
    assert module_2.DEFAULT_CONFIG.formatting_function is None
    assert module_2.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_2.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_2.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_2.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.constants).__module__}.{type(module_2.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.classes).__module__}.{type(module_2.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.variables).__module__}.{type(module_2.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.variables) == 0
    assert module_2.DEFAULT_CONFIG.dedup_headings is False
    assert module_2.DEFAULT_CONFIG.only_sections is False
    assert module_2.DEFAULT_CONFIG.only_modified is False
    assert module_2.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_2.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_2.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_2.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_2.DEFAULT_CONFIG.follow_links is True
    assert module_2.DEFAULT_CONFIG.indented_import_headings is True
    assert module_2.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_2.DEFAULT_CONFIG.reverse_sort is False
    assert module_2.DEFAULT_CONFIG.star_first is False
    assert module_2.DEFAULT_CONFIG.git_ls_files == {}
    assert module_2.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_2.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_2.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_2.DEFAULT_CONFIG.sort_reexports is False
    assert module_2.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_2.Config.known_patterns).__module__}.{type(module_2.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments).__module__}.{type(module_2.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments_end).__module__}.{type(module_2.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skips).__module__}.{type(module_2.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skip_globs).__module__}.{type(module_2.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.sorting_function).__module__}.{type(module_2.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_4 = module_3.maybe(*var_1)
    assert var_4 == '()?'
    assert module_3.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_3.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_3.ENDMARKER == 0
    assert module_3.NAME == 1
    assert module_3.NUMBER == 2
    assert module_3.STRING == 3
    assert module_3.NEWLINE == 4
    assert module_3.INDENT == 5
    assert module_3.DEDENT == 6
    assert module_3.LPAR == 7
    assert module_3.RPAR == 8
    assert module_3.LSQB == 9
    assert module_3.RSQB == 10
    assert module_3.COLON == 11
    assert module_3.COMMA == 12
    assert module_3.SEMI == 13
    assert module_3.PLUS == 14
    assert module_3.MINUS == 15
    assert module_3.STAR == 16
    assert module_3.SLASH == 17
    assert module_3.VBAR == 18
    assert module_3.AMPER == 19
    assert module_3.LESS == 20
    assert module_3.GREATER == 21
    assert module_3.EQUAL == 22
    assert module_3.DOT == 23
    assert module_3.PERCENT == 24
    assert module_3.LBRACE == 25
    assert module_3.RBRACE == 26
    assert module_3.EQEQUAL == 27
    assert module_3.NOTEQUAL == 28
    assert module_3.LESSEQUAL == 29
    assert module_3.GREATEREQUAL == 30
    assert module_3.TILDE == 31
    assert module_3.CIRCUMFLEX == 32
    assert module_3.LEFTSHIFT == 33
    assert module_3.RIGHTSHIFT == 34
    assert module_3.DOUBLESTAR == 35
    assert module_3.PLUSEQUAL == 36
    assert module_3.MINEQUAL == 37
    assert module_3.STAREQUAL == 38
    assert module_3.SLASHEQUAL == 39
    assert module_3.PERCENTEQUAL == 40
    assert module_3.AMPEREQUAL == 41
    assert module_3.VBAREQUAL == 42
    assert module_3.CIRCUMFLEXEQUAL == 43
    assert module_3.LEFTSHIFTEQUAL == 44
    assert module_3.RIGHTSHIFTEQUAL == 45
    assert module_3.DOUBLESTAREQUAL == 46
    assert module_3.DOUBLESLASH == 47
    assert module_3.DOUBLESLASHEQUAL == 48
    assert module_3.AT == 49
    assert module_3.ATEQUAL == 50
    assert module_3.RARROW == 51
    assert module_3.ELLIPSIS == 52
    assert module_3.COLONEQUAL == 53
    assert module_3.OP == 54
    assert module_3.AWAIT == 55
    assert module_3.ASYNC == 56
    assert module_3.TYPE_IGNORE == 57
    assert module_3.TYPE_COMMENT == 58
    assert module_3.SOFT_KEYWORD == 59
    assert module_3.ERRORTOKEN == 60
    assert module_3.COMMENT == 61
    assert module_3.NL == 62
    assert module_3.ENCODING == 63
    assert module_3.N_TOKENS == 64
    assert module_3.NT_OFFSET == 256
    assert module_3.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_3.cookie_re).__module__}.{type(module_3.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_3.blank_re).__module__}.{type(module_3.blank_re).__qualname__}' == 're.Pattern'
    assert module_3.Whitespace == '[ \\f\\t]*'
    assert module_3.Comment == '#[^\\r\\n]*'
    assert module_3.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_3.Name == '\\w+'
    assert module_3.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_3.Binnumber == '0[bB](?:_?[01])+'
    assert module_3.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_3.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_3.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_3.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_3.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_3.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_3.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_3.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_3.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_3.StringPrefix == '(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)'
    assert module_3.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_3.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_3.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_3.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_3.Triple == '((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'\'\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)""")'
    assert module_3.String == '((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_3.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_3.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_3.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_3.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_3.ContStr == '((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_3.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'\'\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"""))'
    assert module_3.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'\'\'|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|Br|rF|bR|Rb|R|Fr|RB|b|fr|FR|rb|U|r|rB|F|rf|RF|fR|f|u|br|Rf|B|BR)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_3.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_3.single_quoted == {"B'", 'F"', 'b"', "Rb'", "R'", 'Br"', 'B"', "fr'", "Br'", "'", 'Fr"', "b'", 'U"', 'RF"', "F'", "br'", "U'", "rb'", "rf'", "Rf'", "f'", 'Rb"', 'bR"', "r'", 'rb"', "bR'", 'fr"', 'RB"', 'rF"', '"', "RB'", 'R"', 'rB"', "fR'", 'br"', "rB'", 'f"', 'Rf"', "Fr'", "BR'", 'FR"', "u'", "RF'", 'u"', 'r"', "rF'", 'fR"', "FR'", 'BR"', 'rf"'}
    assert module_3.triple_quoted == {"'''", "b'''", "br'''", 'u"""', 'F"""', "FR'''", 'f"""', 'br"""', 'rF"""', 'rB"""', "R'''", 'RB"""', 'r"""', "f'''", "B'''", 'U"""', 'BR"""', "bR'''", 'b"""', '"""', "U'''", "fR'''", "u'''", 'rb"""', "rF'''", 'fr"""', 'Rb"""', "rf'''", "Br'''", "Fr'''", 'Rf"""', "rb'''", "rB'''", 'rf"""', "fr'''", "Rf'''", "r'''", 'FR"""', 'R"""', 'fR"""', 'B"""', 'Fr"""', 'bR"""', 'RF"""', "RF'''", "F'''", 'Br"""', "BR'''", "Rb'''", "RB'''"}
    assert module_3.t == 'BR'
    assert module_3.u == "BR'''"
    assert module_3.tabsize == 8
    var_5 = 'force_singlBRe_line'
    var_6 = {var_5: var_4}
    module_0.sort_stream(var_4, var_4, var_2, var_3, var_4, **var_6)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = '",kf-r4M=d\x0bm1_,5'
    var_1 = None
    var_2 = {}
    var_3 = module_0.check_code_string(var_0, var_1, var_0)
    assert var_3 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = {}
    module_0.sort_stream(var_4, var_4, var_0, var_1, var_0, **var_2)

def test_case_10():
    var_0 = None
    var_1 = True
    var_2 = module_0.check_code_string(var_0, var_1)
    assert var_2 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '.RzDo5-%`H;f'
    var_1 = module_0.check_code_string(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = None
    module_0.sort_stream(var_2, var_2, var_2, show_diff=var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.check_code_string(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = module_0.sort_code_string(var_0, var_1, show_diff=var_1)
    assert var_2 == ''
    module_0.check_file(var_2, config=var_0, disregard_skip=var_1, extension=var_0)

def test_case_13():
    var_0 = {}
    var_1 = module_0.find_imports_in_paths(var_0, **var_0)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = list(var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '",kf-r4M=d\x0bm1_,5'
    var_1 = None
    var_2 = module_4.isframe(var_1)
    assert var_2 is False
    assert f'{type(module_4.mod_dict).__module__}.{type(module_4.mod_dict).__qualname__}' == 'builtins.dict'
    assert len(module_4.mod_dict) == 168
    assert module_4.k == 512
    assert module_4.v == 'ASYNC_GENERATOR'
    assert module_4.CO_OPTIMIZED == 1
    assert module_4.CO_NEWLOCALS == 2
    assert module_4.CO_VARARGS == 4
    assert module_4.CO_VARKEYWORDS == 8
    assert module_4.CO_NESTED == 16
    assert module_4.CO_GENERATOR == 32
    assert module_4.CO_NOFREE == 64
    assert module_4.CO_COROUTINE == 128
    assert module_4.CO_ITERABLE_COROUTINE == 256
    assert module_4.CO_ASYNC_GENERATOR == 512
    assert module_4.TPFLAGS_IS_ABSTRACT == 1048576
    assert module_4.modulesbyfile == {'/usr/local/lib/python3.10/importlib/_bootstrap.py': 'importlib._bootstrap', '/usr/local/lib/python3.10/importlib/_bootstrap_external.py': 'importlib._bootstrap_external', '/usr/local/lib/python3.10/codecs.py': 'codecs', '/usr/local/lib/python3.10/encodings/aliases.py': 'encodings.aliases', '/usr/local/lib/python3.10/encodings/__init__.py': 'encodings', '/usr/local/lib/python3.10/encodings/utf_8.py': 'encodings.utf_8', '/usr/local/lib/python3.10/abc.py': 'abc', '/usr/local/lib/python3.10/io.py': 'io', '/usr/local/bin/pynguin': '__main__', '/usr/local/lib/python3.10/stat.py': 'stat', '/usr/local/lib/python3.10/_collections_abc.py': 'collections.abc', '/usr/local/lib/python3.10/genericpath.py': 'genericpath', '/usr/local/lib/python3.10/posixpath.py': 'posixpath', '/usr/local/lib/python3.10/os.py': 'os', '/usr/local/lib/python3.10/_sitebuiltins.py': '_sitebuiltins', '/usr/local/lib/python3.10/__future__.py': '__future__', '/usr/local/lib/python3.10/warnings.py': 'warnings', '/usr/local/lib/python3.10/importlib/__init__.py': 'importlib', '/usr/local/lib/python3.10/importlib/machinery.py': 'importlib.machinery', '/usr/local/lib/python3.10/importlib/_abc.py': 'importlib._abc', '/usr/local/lib/python3.10/keyword.py': 'keyword', '/usr/local/lib/python3.10/operator.py': 'operator', '/usr/local/lib/python3.10/reprlib.py': 'reprlib', '/usr/local/lib/python3.10/collections/__init__.py': 'collections', '/usr/local/lib/python3.10/types.py': 'types', '/usr/local/lib/python3.10/functools.py': 'functools', '/usr/local/lib/python3.10/contextlib.py': 'contextlib', '/usr/local/lib/python3.10/importlib/util.py': 'importlib.util', '/usr/local/lib/python3.10/enum.py': 'enum', '/usr/local/lib/python3.10/sre_constants.py': 'sre_constants', '/usr/local/lib/python3.10/sre_parse.py': 'sre_parse', '/usr/local/lib/python3.10/sre_compile.py': 'sre_compile', '/usr/local/lib/python3.10/copyreg.py': 'copyreg', '/usr/local/lib/python3.10/re.py': 're', '/usr/local/lib/python3.10/fnmatch.py': 'fnmatch', '/usr/local/lib/python3.10/ntpath.py': 'ntpath', '/usr/local/lib/python3.10/urllib/__init__.py': 'urllib', '/usr/local/lib/python3.10/ipaddress.py': 'ipaddress', '/usr/local/lib/python3.10/urllib/parse.py': 'urllib.parse', '/usr/local/lib/python3.10/pathlib.py': 'pathlib', '/usr/local/lib/python3.10/site-packages/__editable___deepmosa_0_1_0_finder.py': '__editable___deepmosa_0_1_0_finder', '/usr/local/lib/python3.10/site-packages/_distutils_hack/__init__.py': '_distutils_hack', '/usr/local/lib/python3.10/site.py': 'site', '/workspace/pynguin/__init__.py': 'pynguin', '/usr/local/lib/python3.10/gettext.py': 'gettext', '/usr/local/lib/python3.10/argparse.py': 'argparse', '/usr/local/lib/python3.10/collections/abc.py': 'collections.abc', '/usr/local/lib/python3.10/concurrent/__init__.py': 'concurrent', '/usr/local/lib/python3.10/token.py': 'token', '/usr/local/lib/python3.10/tokenize.py': 'tokenize', '/usr/local/lib/python3.10/linecache.py': 'linecache', '/usr/local/lib/python3.10/traceback.py': 'traceback', '/usr/local/lib/python3.10/_weakrefset.py': '_weakrefset', '/usr/local/lib/python3.10/weakref.py': 'weakref', '/usr/local/lib/python3.10/string.py': 'string', '/usr/local/lib/python3.10/threading.py': 'threading', '/usr/local/lib/python3.10/logging/__init__.py': 'logging', '/usr/local/lib/python3.10/concurrent/futures/_base.py': 'concurrent.futures._base', '/usr/local/lib/python3.10/concurrent/futures/__init__.py': 'concurrent.futures', '/usr/local/lib/python3.10/lib-dynload/_heapq.cpython-310-x86_64-linux-gnu.so': '_heapq', '/usr/local/lib/python3.10/heapq.py': 'heapq', '/usr/local/lib/python3.10/lib-dynload/_socket.cpython-310-x86_64-linux-gnu.so': '_socket', '/usr/local/lib/python3.10/lib-dynload/math.cpython-310-x86_64-linux-gnu.so': 'math', '/usr/local/lib/python3.10/lib-dynload/select.cpython-310-x86_64-linux-gnu.so': 'select', '/usr/local/lib/python3.10/selectors.py': 'selectors', '/usr/local/lib/python3.10/lib-dynload/array.cpython-310-x86_64-linux-gnu.so': 'array', '/usr/local/lib/python3.10/socket.py': 'socket', '/usr/local/lib/python3.10/signal.py': 'signal', '/usr/local/lib/python3.10/lib-dynload/fcntl.cpython-310-x86_64-linux-gnu.so': 'fcntl', '/usr/local/lib/python3.10/lib-dynload/_posixsubprocess.cpython-310-x86_64-linux-gnu.so': '_posixsubprocess', '/usr/local/lib/python3.10/subprocess.py': 'subprocess', '/usr/local/lib/python3.10/lib-dynload/_ssl.cpython-310-x86_64-linux-gnu.so': '_ssl', '/usr/local/lib/python3.10/lib-dynload/_struct.cpython-310-x86_64-linux-gnu.so': '_struct', '/usr/local/lib/python3.10/struct.py': 'struct', '/usr/local/lib/python3.10/lib-dynload/binascii.cpython-310-x86_64-linux-gnu.so': 'binascii', '/usr/local/lib/python3.10/base64.py': 'base64', '/usr/local/lib/python3.10/ssl.py': 'ssl', '/usr/local/lib/python3.10/asyncio/constants.py': 'asyncio.constants', '/usr/local/lib/python3.10/ast.py': 'ast', '/usr/local/lib/python3.10/lib-dynload/_opcode.cpython-310-x86_64-linux-gnu.so': '_opcode', '/usr/local/lib/python3.10/opcode.py': 'opcode', '/usr/local/lib/python3.10/dis.py': 'dis', '/usr/local/lib/python3.10/inspect.py': 'inspect', '/usr/local/lib/python3.10/asyncio/format_helpers.py': 'asyncio.format_helpers', '/usr/local/lib/python3.10/asyncio/base_futures.py': 'asyncio.base_futures', '/usr/local/lib/python3.10/asyncio/log.py': 'asyncio.log', '/usr/local/lib/python3.10/asyncio/coroutines.py': 'asyncio.coroutines', '/usr/local/lib/python3.10/lib-dynload/_contextvars.cpython-310-x86_64-linux-gnu.so': '_contextvars', '/usr/local/lib/python3.10/contextvars.py': 'contextvars', '/usr/local/lib/python3.10/asyncio/exceptions.py': 'asyncio.exceptions', '/usr/local/lib/python3.10/asyncio/base_tasks.py': 'asyncio.base_tasks', '/usr/local/lib/python3.10/lib-dynload/_asyncio.cpython-310-x86_64-linux-gnu.so': '_asyncio', '/usr/local/lib/python3.10/asyncio/events.py': 'asyncio.events', '/usr/local/lib/python3.10/asyncio/futures.py': 'asyncio.futures', '/usr/local/lib/python3.10/asyncio/protocols.py': 'asyncio.protocols', '/usr/local/lib/python3.10/asyncio/transports.py': 'asyncio.transports', '/usr/local/lib/python3.10/asyncio/sslproto.py': 'asyncio.sslproto', '/usr/local/lib/python3.10/typing.py': 'typing', '/usr/local/lib/python3.10/asyncio/mixins.py': 'asyncio.mixins', '/usr/local/lib/python3.10/asyncio/tasks.py': 'asyncio.tasks', '/usr/local/lib/python3.10/asyncio/locks.py': 'asyncio.locks', '/usr/local/lib/python3.10/asyncio/staggered.py': 'asyncio.staggered', '/usr/local/lib/python3.10/asyncio/trsock.py': 'asyncio.trsock', '/usr/local/lib/python3.10/asyncio/base_events.py': 'asyncio.base_events', '/usr/local/lib/python3.10/asyncio/runners.py': 'asyncio.runners', '/usr/local/lib/python3.10/asyncio/queues.py': 'asyncio.queues', '/usr/local/lib/python3.10/asyncio/streams.py': 'asyncio.streams', '/usr/local/lib/python3.10/asyncio/subprocess.py': 'asyncio.subprocess', '/usr/local/lib/python3.10/asyncio/threads.py': 'asyncio.threads', '/usr/local/lib/python3.10/asyncio/base_subprocess.py': 'asyncio.base_subprocess', '/usr/local/lib/python3.10/asyncio/selector_events.py': 'asyncio.selector_events', '/usr/local/lib/python3.10/asyncio/unix_events.py': 'asyncio.unix_events', '/usr/local/lib/python3.10/asyncio/__init__.py': 'asyncio', '/usr/local/lib/python3.10/lib-dynload/zlib.cpython-310-x86_64-linux-gnu.so': 'zlib', '/usr/local/lib/python3.10/_compression.py': '_compression', '/usr/local/lib/python3.10/lib-dynload/_bz2.cpython-310-x86_64-linux-gnu.so': '_bz2', '/usr/local/lib/python3.10/bz2.py': 'bz2', '/usr/local/lib/python3.10/lib-dynload/_lzma.cpython-310-x86_64-linux-gnu.so': '_lzma', '/usr/local/lib/python3.10/lzma.py': 'lzma', '/usr/local/lib/python3.10/shutil.py': 'shutil', '/usr/local/lib/python3.10/lib-dynload/_bisect.cpython-310-x86_64-linux-gnu.so': '_bisect', '/usr/local/lib/python3.10/bisect.py': 'bisect', '/usr/local/lib/python3.10/lib-dynload/_random.cpython-310-x86_64-linux-gnu.so': '_random', '/usr/local/lib/python3.10/lib-dynload/_sha512.cpython-310-x86_64-linux-gnu.so': '_sha512', '/usr/local/lib/python3.10/random.py': 'random', '/usr/local/lib/python3.10/tempfile.py': 'tempfile', '/usr/local/lib/python3.10/site-packages/dotenv/parser.py': 'dotenv.parser', '/usr/local/lib/python3.10/site-packages/dotenv/variables.py': 'dotenv.variables', '/usr/local/lib/python3.10/site-packages/dotenv/main.py': 'dotenv.main', '/usr/local/lib/python3.10/site-packages/dotenv/__init__.py': 'dotenv', '/usr/local/lib/python3.10/copy.py': 'copy', '/usr/local/lib/python3.10/dataclasses.py': 'dataclasses', '/usr/local/lib/python3.10/site-packages/typing_extensions.py': 'typing_extensions', '/usr/local/lib/python3.10/lib-dynload/_hashlib.cpython-310-x86_64-linux-gnu.so': '_hashlib', '/usr/local/lib/python3.10/lib-dynload/_blake2.cpython-310-x86_64-linux-gnu.so': '_blake2', '/usr/local/lib/python3.10/hashlib.py': 'hashlib', '/usr/local/lib/python3.10/site-packages/simple_parsing/utils.py': 'simple_parsing.utils', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/custom_actions.py': 'simple_parsing.helpers.custom_actions', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/subgroups.py': 'simple_parsing.helpers.subgroups', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/fields.py': 'simple_parsing.helpers.fields', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/flatten.py': 'simple_parsing.helpers.flatten', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/hparams/priors.py': 'simple_parsing.helpers.hparams.priors', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/hparams/hparam.py': 'simple_parsing.helpers.hparams.hparam', '/usr/local/lib/python3.10/_compat_pickle.py': '_compat_pickle', '/usr/local/lib/python3.10/lib-dynload/_pickle.cpython-310-x86_64-linux-gnu.so': '_pickle', '/usr/local/lib/python3.10/pickle.py': 'pickle', '/usr/local/lib/python3.10/site-packages/simple_parsing/annotation_utils/__init__.py': 'simple_parsing.annotation_utils', '/usr/local/lib/python3.10/site-packages/simple_parsing/annotation_utils/get_field_annotations.py': 'simple_parsing.annotation_utils.get_field_annotations', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/serialization/decoding.py': 'simple_parsing.helpers.serialization.decoding', '/usr/local/lib/python3.10/lib-dynload/_json.cpython-310-x86_64-linux-gnu.so': '_json', '/usr/local/lib/python3.10/json/scanner.py': 'json.scanner', '/usr/local/lib/python3.10/json/decoder.py': 'json.decoder', '/usr/local/lib/python3.10/json/encoder.py': 'json.encoder', '/usr/local/lib/python3.10/json/__init__.py': 'json', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/serialization/encoding.py': 'simple_parsing.helpers.serialization.encoding', '/usr/local/lib/python3.10/site-packages/yaml/error.py': 'yaml.error', '/usr/local/lib/python3.10/site-packages/yaml/tokens.py': 'yaml.tokens', '/usr/local/lib/python3.10/site-packages/yaml/events.py': 'yaml.events', '/usr/local/lib/python3.10/site-packages/yaml/nodes.py': 'yaml.nodes', '/usr/local/lib/python3.10/site-packages/yaml/reader.py': 'yaml.reader', '/usr/local/lib/python3.10/site-packages/yaml/scanner.py': 'yaml.scanner', '/usr/local/lib/python3.10/site-packages/yaml/parser.py': 'yaml.parser', '/usr/local/lib/python3.10/site-packages/yaml/composer.py': 'yaml.composer', '/usr/local/lib/python3.10/lib-dynload/_datetime.cpython-310-x86_64-linux-gnu.so': '_datetime', '/usr/local/lib/python3.10/datetime.py': 'datetime', '/usr/local/lib/python3.10/site-packages/yaml/constructor.py': 'yaml.constructor', '/usr/local/lib/python3.10/site-packages/yaml/resolver.py': 'yaml.resolver', '/usr/local/lib/python3.10/site-packages/yaml/loader.py': 'yaml.loader', '/usr/local/lib/python3.10/site-packages/yaml/emitter.py': 'yaml.emitter', '/usr/local/lib/python3.10/site-packages/yaml/serializer.py': 'yaml.serializer', '/usr/local/lib/python3.10/site-packages/yaml/representer.py': 'yaml.representer', '/usr/local/lib/python3.10/site-packages/yaml/dumper.py': 'yaml.dumper', '/usr/local/lib/python3.10/site-packages/yaml/_yaml.cpython-310-x86_64-linux-gnu.so': 'yaml._yaml', '/usr/local/lib/python3.10/site-packages/yaml/cyaml.py': 'yaml.cyaml', '/usr/local/lib/python3.10/site-packages/yaml/__init__.py': 'yaml', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/serialization/serializable.py': 'simple_parsing.helpers.serialization.serializable', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/serialization/yaml_serialization.py': 'simple_parsing.helpers.serialization.yaml_serialization', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/serialization/__init__.py': 'simple_parsing.helpers.serialization', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/hparams/hyperparameters.py': 'simple_parsing.helpers.hparams.hyperparameters', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/hparams/__init__.py': 'simple_parsing.helpers.hparams', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/partial.py': 'simple_parsing.helpers.partial', '/usr/local/lib/python3.10/site-packages/simple_parsing/helpers/__init__.py': 'simple_parsing.helpers', '/usr/local/lib/python3.10/textwrap.py': 'textwrap', '/usr/local/lib/python3.10/site-packages/docstring_parser/common.py': 'docstring_parser.common', '/usr/local/lib/python3.10/site-packages/docstring_parser/epydoc.py': 'docstring_parser.epydoc', '/usr/local/lib/python3.10/site-packages/docstring_parser/google.py': 'docstring_parser.google', '/usr/local/lib/python3.10/site-packages/docstring_parser/numpydoc.py': 'docstring_parser.numpydoc', '/usr/local/lib/python3.10/site-packages/docstring_parser/rest.py': 'docstring_parser.rest', '/usr/local/lib/python3.10/site-packages/docstring_parser/attrdoc.py': 'docstring_parser.attrdoc', '/usr/local/lib/python3.10/site-packages/docstring_parser/parser.py': 'docstring_parser.parser', '/usr/local/lib/python3.10/site-packages/docstring_parser/util.py': 'docstring_parser.util', '/usr/local/lib/python3.10/site-packages/docstring_parser/__init__.py': 'docstring_parser', '/usr/local/lib/python3.10/site-packages/simple_parsing/docstring.py': 'simple_parsing.docstring', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/field_metavar.py': 'simple_parsing.wrappers.field_metavar', '/usr/local/lib/python3.10/site-packages/simple_parsing/help_formatter.py': 'simple_parsing.help_formatter', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/field_parsing.py': 'simple_parsing.wrappers.field_parsing', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/wrapper.py': 'simple_parsing.wrappers.wrapper', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/field_wrapper.py': 'simple_parsing.wrappers.field_wrapper', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/dataclass_wrapper.py': 'simple_parsing.wrappers.dataclass_wrapper', '/usr/local/lib/python3.10/site-packages/simple_parsing/wrappers/__init__.py': 'simple_parsing.wrappers', '/usr/local/lib/python3.10/site-packages/simple_parsing/conflicts.py': 'simple_parsing.conflicts', '/usr/local/lib/python3.10/shlex.py': 'shlex', '/usr/local/lib/python3.10/site-packages/simple_parsing/parsing.py': 'simple_parsing.parsing', '/usr/local/lib/python3.10/site-packages/simple_parsing/decorators.py': 'simple_parsing.decorators', '/usr/local/lib/python3.10/site-packages/simple_parsing/replace.py': 'simple_parsing.replace', '/usr/local/lib/python3.10/site-packages/simple_parsing/__init__.py': 'simple_parsing', '/workspace/pynguin/__version__.py': 'pynguin.__version__', '/workspace/pynguin/utils/__init__.py': 'pynguin.utils', '/workspace/pynguin/utils/statistics/__init__.py': 'pynguin.utils.statistics', '/workspace/pynguin/utils/statistics/runtimevariable.py': 'pynguin.utils.statistics.runtimevariable', '/workspace/pynguin/configuration.py': 'pynguin.configuration', '/workspace/pynguin/cli.py': 'pynguin.cli', '/usr/local/lib/python3.10/locale.py': 'locale', '/workspace/pynguin/assertion/__init__.py': 'pynguin.assertion', '/workspace/pynguin/assertion/assertion.py': 'pynguin.assertion.assertion', '/workspace/pynguin/utils/orderedset.py': 'pynguin.utils.orderedset', '/workspace/pynguin/assertion/assertion_trace.py': 'pynguin.assertion.assertion_trace', '/workspace/pynguin/ga/__init__.py': 'pynguin.ga', '/workspace/pynguin/ga/chromosomevisitor.py': 'pynguin.ga.chromosomevisitor', '/workspace/pynguin/testcase/__init__.py': 'pynguin.testcase', '/usr/local/lib/python3.10/lib-dynload/_queue.cpython-310-x86_64-linux-gnu.so': '_queue', '/usr/local/lib/python3.10/queue.py': 'queue', '/usr/local/lib/python3.10/site-packages/_pytest/_version.py': '_pytest._version', '/usr/local/lib/python3.10/site-packages/_pytest/__init__.py': '_pytest', '/usr/local/lib/python3.10/site-packages/pluggy/_result.py': 'pluggy._result', '/usr/local/lib/python3.10/site-packages/pluggy/_hooks.py': 'pluggy._hooks', '/usr/local/lib/python3.10/site-packages/pluggy/_tracing.py': 'pluggy._tracing', '/usr/local/lib/python3.10/site-packages/pluggy/_warnings.py': 'pluggy._warnings', '/usr/local/lib/python3.10/site-packages/pluggy/_callers.py': 'pluggy._callers', '/usr/local/lib/python3.10/site-packages/pluggy/_manager.py': 'pluggy._manager', '/usr/local/lib/python3.10/site-packages/pluggy/_version.py': 'pluggy._version', '/usr/local/lib/python3.10/site-packages/pluggy/__init__.py': 'pluggy', '/usr/local/lib/python3.10/site-packages/_pytest/_code/source.py': '_pytest._code.source', '/usr/local/lib/python3.10/site-packages/pygments/__init__.py': 'pygments', '/usr/local/lib/python3.10/site-packages/pygments/formatters/_mapping.py': 'pygments.formatters._mapping', '/usr/local/lib/python3.10/lib-dynload/_csv.cpython-310-x86_64-linux-gnu.so': '_csv', '/usr/local/lib/python3.10/csv.py': 'csv', '/usr/local/lib/python3.10/email/__init__.py': 'email', '/usr/local/lib/python3.10/zipfile.py': 'zipfile', '/usr/local/lib/python3.10/uu.py': 'uu', '/usr/local/lib/python3.10/quopri.py': 'quopri', '/usr/local/lib/python3.10/calendar.py': 'calendar', '/usr/local/lib/python3.10/email/_parseaddr.py': 'email._parseaddr', '/usr/local/lib/python3.10/email/base64mime.py': 'email.base64mime', '/usr/local/lib/python3.10/email/quoprimime.py': 'email.quoprimime', '/usr/local/lib/python3.10/email/errors.py': 'email.errors', '/usr/local/lib/python3.10/email/encoders.py': 'email.encoders', '/usr/local/lib/python3.10/email/charset.py': 'email.charset', '/usr/local/lib/python3.10/email/utils.py': 'email.utils', '/usr/local/lib/python3.10/email/header.py': 'email.header', '/usr/local/lib/python3.10/email/_policybase.py': 'email._policybase', '/usr/local/lib/python3.10/email/_encoded_words.py': 'email._encoded_words', '/usr/local/lib/python3.10/email/iterators.py': 'email.iterators', '/usr/local/lib/python3.10/email/message.py': 'email.message', '/usr/local/lib/python3.10/importlib/metadata/_functools.py': 'importlib.metadata._functools', '/usr/local/lib/python3.10/importlib/metadata/_text.py': 'importlib.metadata._text', '/usr/local/lib/python3.10/importlib/metadata/_adapters.py': 'importlib.metadata._adapters', '/usr/local/lib/python3.10/importlib/metadata/_meta.py': 'importlib.metadata._meta', '/usr/local/lib/python3.10/importlib/metadata/_collections.py': 'importlib.metadata._collections', '/usr/local/lib/python3.10/importlib/metadata/_itertools.py': 'importlib.metadata._itertools', '/usr/local/lib/python3.10/importlib/abc.py': 'importlib.abc', '/usr/local/lib/python3.10/importlib/metadata/__init__.py': 'importlib.metadata', '/usr/local/lib/python3.10/site-packages/pygments/plugin.py': 'pygments.plugin', '/usr/local/lib/python3.10/site-packages/pygments/util.py': 'pygments.util', '/usr/local/lib/python3.10/site-packages/pygments/formatters/__init__.py': 'pygments.formatters', '/usr/local/lib/python3.10/site-packages/pygments/styles/_mapping.py': 'pygments.styles._mapping', '/usr/local/lib/python3.10/site-packages/pygments/styles/__init__.py': 'pygments.styles', '/usr/local/lib/python3.10/site-packages/pygments/formatter.py': 'pygments.formatter', '/usr/local/lib/python3.10/site-packages/pygments/token.py': 'pygments.token', '/usr/local/lib/python3.10/site-packages/pygments/console.py': 'pygments.console', '/usr/local/lib/python3.10/site-packages/pygments/formatters/terminal.py': 'pygments.formatters.terminal', '/usr/local/lib/python3.10/site-packages/pygments/filter.py': 'pygments.filter', '/usr/local/lib/python3.10/site-packages/pygments/filters/__init__.py': 'pygments.filters', '/usr/local/lib/python3.10/site-packages/pygments/regexopt.py': 'pygments.regexopt', '/usr/local/lib/python3.10/site-packages/pygments/lexer.py': 'pygments.lexer', '/usr/local/lib/python3.10/site-packages/pygments/lexers/_mapping.py': 'pygments.lexers._mapping', '/usr/local/lib/python3.10/site-packages/pygments/modeline.py': 'pygments.modeline', '/usr/local/lib/python3.10/site-packages/pygments/lexers/__init__.py': 'pygments.lexers', '/usr/local/lib/python3.10/site-packages/pygments/lexers/diff.py': 'pygments.lexers.diff', '/usr/local/lib/python3.10/site-packages/pygments/unistring.py': 'pygments.unistring', '/usr/local/lib/python3.10/site-packages/pygments/lexers/python.py': 'pygments.lexers.python', '/usr/local/lib/python3.10/site-packages/_pytest/_py/__init__.py': '_pytest._py', '/usr/local/lib/python3.10/site-packages/_pytest/_py/error.py': '_pytest._py.error', '/usr/local/lib/python3.10/platform.py': 'platform', '/usr/local/lib/python3.10/lib-dynload/_uuid.cpython-310-x86_64-linux-gnu.so': '_uuid', '/usr/local/lib/python3.10/uuid.py': 'uuid', '/usr/local/lib/python3.10/site-packages/_pytest/_py/path.py': '_pytest._py.path', '/usr/local/lib/python3.10/site-packages/py.py': 'py', '/usr/local/lib/python3.10/site-packages/_pytest/compat.py': '_pytest.compat', '/usr/local/lib/python3.10/lib-dynload/unicodedata.cpython-310-x86_64-linux-gnu.so': 'unicodedata', '/usr/local/lib/python3.10/site-packages/_pytest/_io/wcwidth.py': '_pytest._io.wcwidth', '/usr/local/lib/python3.10/site-packages/_pytest/_io/terminalwriter.py': '_pytest._io.terminalwriter', '/usr/local/lib/python3.10/site-packages/_pytest/_io/__init__.py': '_pytest._io', '/usr/local/lib/python3.10/pprint.py': 'pprint', '/usr/local/lib/python3.10/site-packages/_pytest/_io/saferepr.py': '_pytest._io.saferepr', '/usr/local/lib/python3.10/site-packages/_pytest/warning_types.py': '_pytest.warning_types', '/usr/local/lib/python3.10/site-packages/_pytest/deprecated.py': '_pytest.deprecated', '/usr/local/lib/python3.10/site-packages/_pytest/outcomes.py': '_pytest.outcomes', '/usr/local/lib/python3.10/site-packages/_pytest/pathlib.py': '_pytest.pathlib', '/usr/local/lib/python3.10/site-packages/exceptiongroup/_exceptions.py': 'exceptiongroup._exceptions', '/usr/local/lib/python3.10/site-packages/exceptiongroup/_catch.py': 'exceptiongroup._catch', '/usr/local/lib/python3.10/site-packages/exceptiongroup/_version.py': 'exceptiongroup._version', '/usr/local/lib/python3.10/site-packages/exceptiongroup/_formatting.py': 'exceptiongroup._formatting', '/usr/local/lib/python3.10/site-packages/exceptiongroup/_suppress.py': 'exceptiongroup._suppress', '/usr/local/lib/python3.10/site-packages/exceptiongroup/__init__.py': 'exceptiongroup', '/usr/local/lib/python3.10/site-packages/_pytest/_code/code.py': '_pytest._code.code', '/usr/local/lib/python3.10/site-packages/_pytest/_code/__init__.py': '_pytest._code', '/usr/local/lib/python3.10/importlib/readers.py': 'importlib.readers', '/usr/local/lib/python3.10/site-packages/_pytest/_io/pprint.py': '_pytest._io.pprint', '/usr/local/lib/python3.10/glob.py': 'glob', '/usr/local/lib/python3.10/site-packages/_pytest/config/compat.py': '_pytest.config.compat', '/usr/local/lib/python3.10/site-packages/_pytest/config/exceptions.py': '_pytest.config.exceptions', '/usr/local/lib/python3.10/site-packages/iniconfig/exceptions.py': 'iniconfig.exceptions', '/usr/local/lib/python3.10/site-packages/iniconfig/_parse.py': 'iniconfig._parse', '/usr/local/lib/python3.10/site-packages/iniconfig/__init__.py': 'iniconfig', '/usr/local/lib/python3.10/site-packages/_pytest/config/findpaths.py': '_pytest.config.findpaths', '/usr/local/lib/python3.10/site-packages/_pytest/config/argparsing.py': '_pytest.config.argparsing', '/usr/local/lib/python3.10/site-packages/_pytest/hookspec.py': '_pytest.hookspec', '/usr/local/lib/python3.10/site-packages/_pytest/stash.py': '_pytest.stash', '/usr/local/lib/python3.10/site-packages/_pytest/config/__init__.py': '_pytest.config', '/usr/local/lib/python3.10/site-packages/_pytest/assertion/util.py': '_pytest.assertion.util', '/usr/local/lib/python3.10/site-packages/_pytest/mark/expression.py': '_pytest.mark.expression', '/usr/local/lib/python3.10/site-packages/_pytest/raises.py': '_pytest.raises', '/usr/local/lib/python3.10/site-packages/_pytest/scope.py': '_pytest.scope', '/usr/local/lib/python3.10/site-packages/_pytest/mark/structures.py': '_pytest.mark.structures', '/usr/local/lib/python3.10/site-packages/_pytest/mark/__init__.py': '_pytest.mark', '/usr/local/lib/python3.10/site-packages/_pytest/nodes.py': '_pytest.nodes', '/usr/local/lib/python3.10/site-packages/_pytest/reports.py': '_pytest.reports', '/usr/local/lib/python3.10/bdb.py': 'bdb', '/usr/local/lib/python3.10/site-packages/_pytest/timing.py': '_pytest.timing', '/usr/local/lib/python3.10/site-packages/_pytest/runner.py': '_pytest.runner', '/usr/local/lib/python3.10/site-packages/_pytest/main.py': '_pytest.main', '/usr/local/lib/python3.10/site-packages/_pytest/fixtures.py': '_pytest.fixtures', '/usr/local/lib/python3.10/site-packages/_pytest/assertion/rewrite.py': '_pytest.assertion.rewrite', '/usr/local/lib/python3.10/site-packages/_pytest/assertion/truncate.py': '_pytest.assertion.truncate', '/usr/local/lib/python3.10/site-packages/_pytest/assertion/__init__.py': '_pytest.assertion', '/usr/local/lib/python3.10/site-packages/_pytest/cacheprovider.py': '_pytest.cacheprovider', '/usr/local/lib/python3.10/site-packages/_pytest/capture.py': '_pytest.capture', '/usr/local/lib/python3.10/unittest/util.py': 'unittest.util', '/usr/local/lib/python3.10/unittest/result.py': 'unittest.result', '/usr/local/lib/python3.10/difflib.py': 'difflib', '/usr/local/lib/python3.10/unittest/case.py': 'unittest.case', '/usr/local/lib/python3.10/unittest/suite.py': 'unittest.suite', '/usr/local/lib/python3.10/unittest/loader.py': 'unittest.loader', '/usr/local/lib/python3.10/unittest/signals.py': 'unittest.signals', '/usr/local/lib/python3.10/unittest/runner.py': 'unittest.runner', '/usr/local/lib/python3.10/unittest/main.py': 'unittest.main', '/usr/local/lib/python3.10/unittest/__init__.py': 'unittest', '/usr/local/lib/python3.10/site-packages/_pytest/debugging.py': '_pytest.debugging', '/usr/local/lib/python3.10/site-packages/_pytest/python.py': '_pytest.python', '/usr/local/lib/python3.10/numbers.py': 'numbers', '/usr/local/lib/python3.10/lib-dynload/_decimal.cpython-310-x86_64-linux-gnu.so': 'decimal', '/usr/local/lib/python3.10/decimal.py': 'decimal', '/usr/local/lib/python3.10/site-packages/_pytest/python_api.py': '_pytest.python_api', '/usr/local/lib/python3.10/site-packages/_pytest/doctest.py': '_pytest.doctest', '/usr/local/lib/python3.10/site-packages/_pytest/freeze_support.py': '_pytest.freeze_support', '/usr/local/lib/python3.10/site-packages/_pytest/monkeypatch.py': '_pytest.monkeypatch', '/usr/local/lib/python3.10/site-packages/_pytest/tmpdir.py': '_pytest.tmpdir', '/usr/local/lib/python3.10/site-packages/_pytest/pytester.py': '_pytest.pytester', '/usr/local/lib/python3.10/site-packages/_pytest/terminal.py': '_pytest.terminal', '/usr/local/lib/python3.10/site-packages/_pytest/legacypath.py': '_pytest.legacypath', '/usr/local/lib/python3.10/site-packages/_pytest/logging.py': '_pytest.logging', '/usr/local/lib/python3.10/site-packages/_pytest/recwarn.py': '_pytest.recwarn', '/usr/local/lib/python3.10/site-packages/_pytest/subtests.py': '_pytest.subtests', '/usr/local/lib/python3.10/site-packages/pytest/__init__.py': 'pytest', '/usr/local/lib/python3.10/site-packages/bytecode/utils.py': 'bytecode.utils', '/usr/local/lib/python3.10/site-packages/bytecode/instr.py': 'bytecode.instr', '/usr/local/lib/python3.10/site-packages/bytecode/flags.py': 'bytecode.flags', '/usr/local/lib/python3.10/site-packages/bytecode/bytecode.py': 'bytecode.bytecode', '/usr/local/lib/python3.10/site-packages/bytecode/concrete.py': 'bytecode.concrete', '/usr/local/lib/python3.10/site-packages/bytecode/cfg.py': 'bytecode.cfg', '/usr/local/lib/python3.10/site-packages/bytecode/version.py': 'bytecode.version', '/usr/local/lib/python3.10/site-packages/bytecode/__init__.py': 'bytecode', '/usr/local/lib/python3.10/site-packages/jellyfish/_rustyfish.cpython-310-x86_64-linux-gnu.so': 'jellyfish._rustyfish', '/usr/local/lib/python3.10/site-packages/jellyfish/_jellyfish.py': 'jellyfish._jellyfish', '/usr/local/lib/python3.10/site-packages/jellyfish/__init__.py': 'jellyfish', '/workspace/pynguin/analyses/__init__.py': 'pynguin.analyses', '/usr/local/lib/python3.10/site-packages/networkx/lazy_imports.py': 'networkx.lazy_imports', '/usr/local/lib/python3.10/site-packages/networkx/exception.py': 'networkx.exception', '/usr/local/lib/python3.10/site-packages/networkx/utils/misc.py': 'networkx.utils.misc', '/usr/local/lib/python3.10/gzip.py': 'gzip', '/usr/local/lib/python3.10/site-packages/networkx/utils/decorators.py': 'networkx.utils.decorators', '/usr/local/lib/python3.10/site-packages/networkx/utils/random_sequence.py': 'networkx.utils.random_sequence', '/usr/local/lib/python3.10/site-packages/networkx/utils/union_find.py': 'networkx.utils.union_find', '/usr/local/lib/python3.10/site-packages/networkx/utils/rcm.py': 'networkx.utils.rcm', '/usr/local/lib/python3.10/site-packages/networkx/utils/heaps.py': 'networkx.utils.heaps', '/usr/local/lib/python3.10/site-packages/networkx/utils/configs.py': 'networkx.utils.configs', '/usr/local/lib/python3.10/site-packages/networkx/utils/backends.py': 'networkx.utils.backends', '/usr/local/lib/python3.10/site-packages/networkx/utils/__init__.py': 'networkx.utils', '/usr/local/lib/python3.10/site-packages/networkx/convert.py': 'networkx.convert', '/usr/local/lib/python3.10/site-packages/networkx/classes/coreviews.py': 'networkx.classes.coreviews', '/usr/local/lib/python3.10/site-packages/networkx/classes/reportviews.py': 'networkx.classes.reportviews', '/usr/local/lib/python3.10/site-packages/networkx/classes/graph.py': 'networkx.classes.graph', '/usr/local/lib/python3.10/site-packages/networkx/classes/digraph.py': 'networkx.classes.digraph', '/usr/local/lib/python3.10/site-packages/networkx/classes/multigraph.py': 'networkx.classes.multigraph', '/usr/local/lib/python3.10/site-packages/networkx/classes/multidigraph.py': 'networkx.classes.multidigraph', '/usr/local/lib/python3.10/site-packages/networkx/classes/function.py': 'networkx.classes.function', '/usr/local/lib/python3.10/site-packages/networkx/classes/filters.py': 'networkx.classes.filters', '/usr/local/lib/python3.10/site-packages/networkx/classes/graphviews.py': 'networkx.classes.graphviews', '/usr/local/lib/python3.10/site-packages/networkx/classes/__init__.py': 'networkx.classes', '/usr/local/lib/python3.10/site-packages/networkx/convert_matrix.py': 'networkx.convert_matrix', '/usr/local/lib/python3.10/site-packages/networkx/relabel.py': 'networkx.relabel', '/usr/local/lib/python3.10/importlib/_adapters.py': 'importlib._adapters', '/usr/local/lib/python3.10/importlib/_common.py': 'importlib._common', '/usr/local/lib/python3.10/importlib/resources.py': 'importlib.resources', '/usr/local/lib/python3.10/site-packages/networkx/generators/atlas.py': 'networkx.generators.atlas', '/usr/local/lib/python3.10/site-packages/networkx/generators/classic.py': 'networkx.generators.classic', '/usr/local/lib/python3.10/site-packages/networkx/generators/cographs.py': 'networkx.generators.cographs', '/usr/local/lib/python3.10/site-packages/networkx/generators/community.py': 'networkx.generators.community', '/usr/local/lib/python3.10/site-packages/networkx/generators/degree_seq.py': 'networkx.generators.degree_seq', '/usr/local/lib/python3.10/site-packages/networkx/generators/directed.py': 'networkx.generators.directed', '/usr/local/lib/python3.10/site-packages/networkx/generators/duplication.py': 'networkx.generators.duplication', '/usr/local/lib/python3.10/site-packages/networkx/generators/ego.py': 'networkx.generators.ego', '/usr/local/lib/python3.10/site-packages/networkx/generators/expanders.py': 'networkx.generators.expanders', '/usr/local/lib/python3.10/site-packages/networkx/generators/geometric.py': 'networkx.generators.geometric', '/usr/local/lib/python3.10/site-packages/networkx/generators/harary_graph.py': 'networkx.generators.harary_graph', '/usr/local/lib/python3.10/site-packages/networkx/generators/internet_as_graphs.py': 'networkx.generators.internet_as_graphs', '/usr/local/lib/python3.10/site-packages/networkx/generators/intersection.py': 'networkx.generators.intersection', '/usr/local/lib/python3.10/site-packages/networkx/generators/interval_graph.py': 'networkx.generators.interval_graph', '/usr/local/lib/python3.10/site-packages/networkx/generators/joint_degree_seq.py': 'networkx.generators.joint_degree_seq', '/usr/local/lib/python3.10/site-packages/networkx/generators/lattice.py': 'networkx.generators.lattice', '/usr/local/lib/python3.10/site-packages/networkx/generators/line.py': 'networkx.generators.line', '/usr/local/lib/python3.10/site-packages/networkx/generators/mycielski.py': 'networkx.generators.mycielski', '/usr/local/lib/python3.10/site-packages/networkx/generators/nonisomorphic_trees.py': 'networkx.generators.nonisomorphic_trees', '/usr/local/lib/python3.10/site-packages/networkx/generators/random_clustered.py': 'networkx.generators.random_clustered', '/usr/local/lib/python3.10/site-packages/networkx/generators/random_graphs.py': 'networkx.generators.random_graphs', '/usr/local/lib/python3.10/site-packages/networkx/generators/small.py': 'networkx.generators.small', '/usr/local/lib/python3.10/site-packages/networkx/generators/social.py': 'networkx.generators.social', '/usr/local/lib/python3.10/site-packages/networkx/generators/spectral_graph_forge.py': 'networkx.generators.spectral_graph_forge', '/usr/local/lib/python3.10/site-packages/networkx/generators/stochastic.py': 'networkx.generators.stochastic', '/usr/local/lib/python3.10/site-packages/networkx/generators/sudoku.py': 'networkx.generators.sudoku', '/usr/local/lib/python3.10/site-packages/networkx/generators/time_series.py': 'networkx.generators.time_series', '/usr/local/lib/python3.10/site-packages/networkx/generators/trees.py': 'networkx.generators.trees', '/usr/local/lib/python3.10/site-packages/networkx/generators/triads.py': 'networkx.generators.triads', '/usr/local/lib/python3.10/site-packages/networkx/generators/__init__.py': 'networkx.generators', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/adjlist.py': 'networkx.readwrite.adjlist', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/multiline_adjlist.py': 'networkx.readwrite.multiline_adjlist', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/edgelist.py': 'networkx.readwrite.edgelist', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/pajek.py': 'networkx.readwrite.pajek', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/leda.py': 'networkx.readwrite.leda', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/graph6.py': 'networkx.readwrite.graph6', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/sparse6.py': 'networkx.readwrite.sparse6', '/usr/local/lib/python3.10/html/entities.py': 'html.entities', '/usr/local/lib/python3.10/html/__init__.py': 'html', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/gml.py': 'networkx.readwrite.gml', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/graphml.py': 'networkx.readwrite.graphml', '/usr/local/lib/python3.10/xml/__init__.py': 'xml', '/usr/local/lib/python3.10/xml/etree/__init__.py': 'xml.etree', '/usr/local/lib/python3.10/xml/etree/ElementPath.py': 'xml.etree.ElementPath', '/usr/local/lib/python3.10/lib-dynload/pyexpat.cpython-310-x86_64-linux-gnu.so': 'pyexpat', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/gexf.py': 'networkx.readwrite.gexf', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/json_graph/node_link.py': 'networkx.readwrite.json_graph.node_link', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/json_graph/adjacency.py': 'networkx.readwrite.json_graph.adjacency', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/json_graph/tree.py': 'networkx.readwrite.json_graph.tree', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/json_graph/cytoscape.py': 'networkx.readwrite.json_graph.cytoscape', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/json_graph/__init__.py': 'networkx.readwrite.json_graph', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/text.py': 'networkx.readwrite.text', '/usr/local/lib/python3.10/site-packages/networkx/readwrite/__init__.py': 'networkx.readwrite', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/connectivity.py': 'networkx.algorithms.assortativity.connectivity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/pairs.py': 'networkx.algorithms.assortativity.pairs', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/mixing.py': 'networkx.algorithms.assortativity.mixing', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/correlation.py': 'networkx.algorithms.assortativity.correlation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/neighbor_degree.py': 'networkx.algorithms.assortativity.neighbor_degree', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/assortativity/__init__.py': 'networkx.algorithms.assortativity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/asteroidal.py': 'networkx.algorithms.asteroidal', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/boundary.py': 'networkx.algorithms.boundary', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/broadcasting.py': 'networkx.algorithms.broadcasting', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bridges.py': 'networkx.algorithms.bridges', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/chains.py': 'networkx.algorithms.chains', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/generic.py': 'networkx.algorithms.shortest_paths.generic', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/unweighted.py': 'networkx.algorithms.shortest_paths.unweighted', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/weighted.py': 'networkx.algorithms.shortest_paths.weighted', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/astar.py': 'networkx.algorithms.shortest_paths.astar', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/dense.py': 'networkx.algorithms.shortest_paths.dense', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/shortest_paths/__init__.py': 'networkx.algorithms.shortest_paths', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/betweenness.py': 'networkx.algorithms.centrality.betweenness', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/betweenness_subset.py': 'networkx.algorithms.centrality.betweenness_subset', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/closeness.py': 'networkx.algorithms.centrality.closeness', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/flow_matrix.py': 'networkx.algorithms.centrality.flow_matrix', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/current_flow_betweenness.py': 'networkx.algorithms.centrality.current_flow_betweenness', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/current_flow_betweenness_subset.py': 'networkx.algorithms.centrality.current_flow_betweenness_subset', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/current_flow_closeness.py': 'networkx.algorithms.centrality.current_flow_closeness', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/degree_alg.py': 'networkx.algorithms.centrality.degree_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/dispersion.py': 'networkx.algorithms.centrality.dispersion', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/eigenvector.py': 'networkx.algorithms.centrality.eigenvector', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/group.py': 'networkx.algorithms.centrality.group', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/harmonic.py': 'networkx.algorithms.centrality.harmonic', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/katz.py': 'networkx.algorithms.centrality.katz', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/load.py': 'networkx.algorithms.centrality.load', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/percolation.py': 'networkx.algorithms.centrality.percolation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/reaching.py': 'networkx.algorithms.centrality.reaching', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/second_order.py': 'networkx.algorithms.centrality.second_order', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/subgraph_alg.py': 'networkx.algorithms.centrality.subgraph_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/trophic.py': 'networkx.algorithms.centrality.trophic', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/voterank_alg.py': 'networkx.algorithms.centrality.voterank_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/laplacian.py': 'networkx.algorithms.centrality.laplacian', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/centrality/__init__.py': 'networkx.algorithms.centrality', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/connected.py': 'networkx.algorithms.components.connected', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/strongly_connected.py': 'networkx.algorithms.components.strongly_connected', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/weakly_connected.py': 'networkx.algorithms.components.weakly_connected', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/attracting.py': 'networkx.algorithms.components.attracting', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/biconnected.py': 'networkx.algorithms.components.biconnected', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/semiconnected.py': 'networkx.algorithms.components.semiconnected', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/components/__init__.py': 'networkx.algorithms.components', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/chordal.py': 'networkx.algorithms.chordal', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/cluster.py': 'networkx.algorithms.cluster', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/clique.py': 'networkx.algorithms.clique', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/communicability_alg.py': 'networkx.algorithms.communicability_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/coloring/greedy_coloring.py': 'networkx.algorithms.coloring.greedy_coloring', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/coloring/equitable_coloring.py': 'networkx.algorithms.coloring.equitable_coloring', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/coloring/__init__.py': 'networkx.algorithms.coloring', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/core.py': 'networkx.algorithms.core', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/covering.py': 'networkx.algorithms.covering', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/cycles.py': 'networkx.algorithms.cycles', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/cuts.py': 'networkx.algorithms.cuts', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/d_separation.py': 'networkx.algorithms.d_separation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/dag.py': 'networkx.algorithms.dag', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/distance_measures.py': 'networkx.algorithms.distance_measures', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/distance_regular.py': 'networkx.algorithms.distance_regular', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/dominance.py': 'networkx.algorithms.dominance', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/dominating.py': 'networkx.algorithms.dominating', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/efficiency_measures.py': 'networkx.algorithms.efficiency_measures', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/euler.py': 'networkx.algorithms.euler', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/graphical.py': 'networkx.algorithms.graphical', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/hierarchy.py': 'networkx.algorithms.hierarchy', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/hybrid.py': 'networkx.algorithms.hybrid', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/link_analysis/hits_alg.py': 'networkx.algorithms.link_analysis.hits_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/link_analysis/pagerank_alg.py': 'networkx.algorithms.link_analysis.pagerank_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/link_analysis/__init__.py': 'networkx.algorithms.link_analysis', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/link_prediction.py': 'networkx.algorithms.link_prediction', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/lowest_common_ancestors.py': 'networkx.algorithms.lowest_common_ancestors', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isolate.py': 'networkx.algorithms.isolate', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/matching.py': 'networkx.algorithms.matching', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/minors/contraction.py': 'networkx.algorithms.minors.contraction', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/minors/__init__.py': 'networkx.algorithms.minors', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/mis.py': 'networkx.algorithms.mis', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/moral.py': 'networkx.algorithms.moral', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/non_randomness.py': 'networkx.algorithms.non_randomness', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/operators/all.py': 'networkx.algorithms.operators.all', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/operators/binary.py': 'networkx.algorithms.operators.binary', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/operators/product.py': 'networkx.algorithms.operators.product', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/operators/unary.py': 'networkx.algorithms.operators.unary', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/operators/__init__.py': 'networkx.algorithms.operators', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/planarity.py': 'networkx.algorithms.planarity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/planar_drawing.py': 'networkx.algorithms.planar_drawing', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/polynomials.py': 'networkx.algorithms.polynomials', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/reciprocity.py': 'networkx.algorithms.reciprocity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/regular.py': 'networkx.algorithms.regular', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/richclub.py': 'networkx.algorithms.richclub', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/similarity.py': 'networkx.algorithms.similarity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/graph_hashing.py': 'networkx.algorithms.graph_hashing', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/simple_paths.py': 'networkx.algorithms.simple_paths', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/smallworld.py': 'networkx.algorithms.smallworld', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/smetric.py': 'networkx.algorithms.smetric', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/structuralholes.py': 'networkx.algorithms.structuralholes', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/sparsifiers.py': 'networkx.algorithms.sparsifiers', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/summarization.py': 'networkx.algorithms.summarization', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/swap.py': 'networkx.algorithms.swap', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/time_dependent.py': 'networkx.algorithms.time_dependent', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/beamsearch.py': 'networkx.algorithms.traversal.beamsearch', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/breadth_first_search.py': 'networkx.algorithms.traversal.breadth_first_search', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/depth_first_search.py': 'networkx.algorithms.traversal.depth_first_search', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/edgedfs.py': 'networkx.algorithms.traversal.edgedfs', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/edgebfs.py': 'networkx.algorithms.traversal.edgebfs', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/traversal/__init__.py': 'networkx.algorithms.traversal', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/triads.py': 'networkx.algorithms.triads', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/vitality.py': 'networkx.algorithms.vitality', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/voronoi.py': 'networkx.algorithms.voronoi', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/walks.py': 'networkx.algorithms.walks', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/wiener.py': 'networkx.algorithms.wiener', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/clustering_coefficient.py': 'networkx.algorithms.approximation.clustering_coefficient', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/ramsey.py': 'networkx.algorithms.approximation.ramsey', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/clique.py': 'networkx.algorithms.approximation.clique', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/connectivity.py': 'networkx.algorithms.approximation.connectivity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/distance_measures.py': 'networkx.algorithms.approximation.distance_measures', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/dominating_set.py': 'networkx.algorithms.approximation.dominating_set', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/kcomponents.py': 'networkx.algorithms.approximation.kcomponents', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/matching.py': 'networkx.algorithms.approximation.matching', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/steinertree.py': 'networkx.algorithms.approximation.steinertree', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/recognition.py': 'networkx.algorithms.tree.recognition', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/branchings.py': 'networkx.algorithms.tree.branchings', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/coding.py': 'networkx.algorithms.tree.coding', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/mst.py': 'networkx.algorithms.tree.mst', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/operations.py': 'networkx.algorithms.tree.operations', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/decomposition.py': 'networkx.algorithms.tree.decomposition', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tree/__init__.py': 'networkx.algorithms.tree', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/traveling_salesman.py': 'networkx.algorithms.approximation.traveling_salesman', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/treewidth.py': 'networkx.algorithms.approximation.treewidth', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/vertex_cover.py': 'networkx.algorithms.approximation.vertex_cover', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/maxcut.py': 'networkx.algorithms.approximation.maxcut', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/approximation/__init__.py': 'networkx.algorithms.approximation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/basic.py': 'networkx.algorithms.bipartite.basic', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/centrality.py': 'networkx.algorithms.bipartite.centrality', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/cluster.py': 'networkx.algorithms.bipartite.cluster', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/matrix.py': 'networkx.algorithms.bipartite.matrix', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/matching.py': 'networkx.algorithms.bipartite.matching', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/covering.py': 'networkx.algorithms.bipartite.covering', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/edgelist.py': 'networkx.algorithms.bipartite.edgelist', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/projection.py': 'networkx.algorithms.bipartite.projection', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/redundancy.py': 'networkx.algorithms.bipartite.redundancy', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/spectral.py': 'networkx.algorithms.bipartite.spectral', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/generators.py': 'networkx.algorithms.bipartite.generators', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/extendability.py': 'networkx.algorithms.bipartite.extendability', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/bipartite/__init__.py': 'networkx.algorithms.bipartite', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/node_classification.py': 'networkx.algorithms.node_classification', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/asyn_fluid.py': 'networkx.algorithms.community.asyn_fluid', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/centrality.py': 'networkx.algorithms.community.centrality', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/divisive.py': 'networkx.algorithms.community.divisive', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/kclique.py': 'networkx.algorithms.community.kclique', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/community_utils.py': 'networkx.algorithms.community.community_utils', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/kernighan_lin.py': 'networkx.algorithms.community.kernighan_lin', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/label_propagation.py': 'networkx.algorithms.community.label_propagation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/lukes.py': 'networkx.algorithms.community.lukes', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/quality.py': 'networkx.algorithms.community.quality', '/usr/local/lib/python3.10/site-packages/networkx/utils/mapped_queue.py': 'networkx.utils.mapped_queue', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/modularity_max.py': 'networkx.algorithms.community.modularity_max', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/louvain.py': 'networkx.algorithms.community.louvain', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/community/__init__.py': 'networkx.algorithms.community', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/utils.py': 'networkx.algorithms.flow.utils', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/boykovkolmogorov.py': 'networkx.algorithms.flow.boykovkolmogorov', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/dinitz_alg.py': 'networkx.algorithms.flow.dinitz_alg', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/edmondskarp.py': 'networkx.algorithms.flow.edmondskarp', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/preflowpush.py': 'networkx.algorithms.flow.preflowpush', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/shortestaugmentingpath.py': 'networkx.algorithms.flow.shortestaugmentingpath', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/maxflow.py': 'networkx.algorithms.flow.maxflow', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/mincost.py': 'networkx.algorithms.flow.mincost', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/gomory_hu.py': 'networkx.algorithms.flow.gomory_hu', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/capacityscaling.py': 'networkx.algorithms.flow.capacityscaling', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/networksimplex.py': 'networkx.algorithms.flow.networksimplex', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/flow/__init__.py': 'networkx.algorithms.flow', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/isomorph.py': 'networkx.algorithms.isomorphism.isomorph', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/isomorphvf2.py': 'networkx.algorithms.isomorphism.isomorphvf2', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/vf2userfunc.py': 'networkx.algorithms.isomorphism.vf2userfunc', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/matchhelpers.py': 'networkx.algorithms.isomorphism.matchhelpers', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/temporalisomorphvf2.py': 'networkx.algorithms.isomorphism.temporalisomorphvf2', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/ismags.py': 'networkx.algorithms.isomorphism.ismags', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/tree_isomorphism.py': 'networkx.algorithms.isomorphism.tree_isomorphism', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/vf2pp.py': 'networkx.algorithms.isomorphism.vf2pp', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/isomorphism/__init__.py': 'networkx.algorithms.isomorphism', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/tournament.py': 'networkx.algorithms.tournament', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/utils.py': 'networkx.algorithms.connectivity.utils', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/connectivity.py': 'networkx.algorithms.connectivity.connectivity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/cuts.py': 'networkx.algorithms.connectivity.cuts', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/edge_augmentation.py': 'networkx.algorithms.connectivity.edge_augmentation', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/edge_kcomponents.py': 'networkx.algorithms.connectivity.edge_kcomponents', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/disjoint_paths.py': 'networkx.algorithms.connectivity.disjoint_paths', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/kcomponents.py': 'networkx.algorithms.connectivity.kcomponents', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/kcutsets.py': 'networkx.algorithms.connectivity.kcutsets', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/stoerwagner.py': 'networkx.algorithms.connectivity.stoerwagner', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/connectivity/__init__.py': 'networkx.algorithms.connectivity', '/usr/local/lib/python3.10/site-packages/networkx/algorithms/__init__.py': 'networkx.algorithms', '/usr/local/lib/python3.10/site-packages/networkx/linalg/attrmatrix.py': 'networkx.linalg.attrmatrix', '/usr/local/lib/python3.10/site-packages/networkx/linalg/spectrum.py': 'networkx.linalg.spectrum', '/usr/local/lib/python3.10/site-packages/networkx/linalg/graphmatrix.py': 'networkx.linalg.graphmatrix', '/usr/local/lib/python3.10/site-packages/networkx/linalg/laplacianmatrix.py': 'networkx.linalg.laplacianmatrix', '/usr/local/lib/python3.10/site-packages/networkx/linalg/algebraicconnectivity.py': 'networkx.linalg.algebraicconnectivity', '/usr/local/lib/python3.10/site-packages/networkx/linalg/modularitymatrix.py': 'networkx.linalg.modularitymatrix', '/usr/local/lib/python3.10/site-packages/networkx/linalg/bethehessianmatrix.py': 'networkx.linalg.bethehessianmatrix', '/usr/local/lib/python3.10/site-packages/networkx/linalg/__init__.py': 'networkx.linalg', '/usr/local/lib/python3.10/site-packages/networkx/drawing/layout.py': 'networkx.drawing.layout', '/usr/local/lib/python3.10/site-packages/networkx/drawing/nx_latex.py': 'networkx.drawing.nx_latex', '/usr/local/lib/python3.10/site-packages/networkx/drawing/nx_pylab.py': 'networkx.drawing.nx_pylab', '/usr/local/lib/python3.10/site-packages/networkx/drawing/nx_agraph.py': 'networkx.drawing.nx_agraph', '/usr/local/lib/python3.10/site-packages/networkx/drawing/nx_pydot.py': 'networkx.drawing.nx_pydot', '/usr/local/lib/python3.10/site-packages/networkx/drawing/__init__.py': 'networkx.drawing', '/usr/local/lib/python3.10/site-packages/networkx/__init__.py': 'networkx', '/usr/local/lib/python3.10/site-packages/mypy_extensions.py': 'mypy_extensions', '/usr/local/lib/python3.10/site-packages/typing_inspect.py': 'typing_inspect', '/usr/local/lib/python3.10/site-packages/asciitree/util.py': 'asciitree.util', '/usr/local/lib/python3.10/site-packages/asciitree/drawing.py': 'asciitree.drawing', '/usr/local/lib/python3.10/site-packages/asciitree/traversal.py': 'asciitree.traversal', '/usr/local/lib/python3.10/site-packages/asciitree/__init__.py': 'asciitree', '/workspace/libs/__init__.py': 'libs', '/usr/local/lib/python3.10/site-packages/colorama/ansi.py': 'colorama.ansi', '/usr/local/lib/python3.10/lib-dynload/_ctypes.cpython-310-x86_64-linux-gnu.so': '_ctypes', '/usr/local/lib/python3.10/ctypes/_endian.py': 'ctypes._endian', '/usr/local/lib/python3.10/ctypes/__init__.py': 'ctypes', '/usr/local/lib/python3.10/site-packages/colorama/win32.py': 'colorama.win32', '/usr/local/lib/python3.10/site-packages/colorama/winterm.py': 'colorama.winterm', '/usr/local/lib/python3.10/site-packages/colorama/ansitowin32.py': 'colorama.ansitowin32', '/usr/local/lib/python3.10/site-packages/colorama/initialise.py': 'colorama.initialise', '/usr/local/lib/python3.10/site-packages/colorama/__init__.py': 'colorama', '/workspace/libs/custom_logger/formatter.py': 'libs.custom_logger.formatter', '/workspace/libs/custom_logger/logger.py': 'libs.custom_logger.logger', '/workspace/libs/custom_logger/__init__.py': 'libs.custom_logger', '/workspace/pynguin/utils/typetracing.py': 'pynguin.utils.typetracing', '/workspace/pynguin/utils/randomness.py': 'pynguin.utils.randomness', '/workspace/pynguin/utils/exceptions.py': 'pynguin.utils.exceptions', '/workspace/pynguin/utils/type_utils.py': 'pynguin.utils.type_utils', '/workspace/pynguin/analyses/typesystem.py': 'pynguin.analyses.typesystem', '/workspace/pynguin/testcase/variablereference.py': 'pynguin.testcase.variablereference', '/workspace/pynguin/utils/ast_util.py': 'pynguin.utils.ast_util', '/workspace/pynguin/utils/namingscope.py': 'pynguin.utils.namingscope', '/workspace/pynguin/assertion/assertion_to_ast.py': 'pynguin.assertion.assertion_to_ast', '/workspace/pynguin/slicer/__init__.py': 'pynguin.slicer', '/workspace/pynguin/slicer/executedinstruction.py': 'pynguin.slicer.executedinstruction', '/workspace/pynguin/utils/generic/genericaccessibleobject.py': 'pynguin.utils.generic.genericaccessibleobject', '/workspace/pynguin/utils/generic/__init__.py': 'pynguin.utils.generic', '/usr/local/lib/python3.10/pkgutil.py': 'pkgutil', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/__init__.py': 'distutils', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/recipes.py': 'more_itertools.recipes', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/more.py': 'more_itertools.more', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/more_itertools/__init__.py': 'more_itertools', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/jaraco/functools/__init__.py': 'jaraco.functools', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/compat/__init__.py': 'distutils.compat', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/compat/py39.py': 'distutils.compat.py39', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/compilers/C/errors.py': 'distutils.compilers.C.errors', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/errors.py': 'distutils.errors', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/_modified.py': 'distutils._modified', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/_log.py': 'distutils._log', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/file_util.py': 'distutils.file_util', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/dir_util.py': 'distutils.dir_util', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/debug.py': 'distutils.debug', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/spawn.py': 'distutils.spawn', '/usr/local/lib/python3.10/lib-dynload/grp.cpython-310-x86_64-linux-gnu.so': 'grp', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/archive_util.py': 'distutils.archive_util', '/usr/local/lib/python3.10/sysconfig.py': 'sysconfig', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/util.py': 'distutils.util', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/cmd.py': 'distutils.cmd', '/usr/local/lib/python3.10/site-packages/packaging/__init__.py': 'packaging', '/usr/local/lib/python3.10/site-packages/packaging/_elffile.py': 'packaging._elffile', '/usr/local/lib/python3.10/site-packages/packaging/_manylinux.py': 'packaging._manylinux', '/usr/local/lib/python3.10/site-packages/packaging/_musllinux.py': 'packaging._musllinux', '/usr/local/lib/python3.10/site-packages/packaging/tags.py': 'packaging.tags', '/usr/local/lib/python3.10/site-packages/packaging/version.py': 'packaging.version', '/usr/local/lib/python3.10/site-packages/packaging/utils.py': 'packaging.utils', '/usr/local/lib/python3.10/getopt.py': 'getopt', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/fancy_getopt.py': 'distutils.fancy_getopt', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/dist.py': 'distutils.dist', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/extension.py': 'distutils.extension', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/core.py': 'distutils.core', '/usr/local/lib/python3.10/site-packages/_distutils_hack/override.py': '_distutils_hack.override', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/filelist.py': 'distutils.filelist', '/usr/local/lib/python3.10/site-packages/setuptools/monkey.py': 'setuptools.monkey', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/log.py': 'distutils.log', '/usr/local/lib/python3.10/site-packages/setuptools/logging.py': 'setuptools.logging', '/usr/local/lib/python3.10/site-packages/setuptools/_imp.py': 'setuptools._imp', '/usr/local/lib/python3.10/site-packages/setuptools/depends.py': 'setuptools.depends', '/usr/local/lib/python3.10/site-packages/setuptools/_path.py': 'setuptools._path', '/usr/local/lib/python3.10/site-packages/setuptools/discovery.py': 'setuptools.discovery', '/usr/local/lib/python3.10/site-packages/packaging/specifiers.py': 'packaging.specifiers', '/usr/local/lib/python3.10/site-packages/packaging/_tokenizer.py': 'packaging._tokenizer', '/usr/local/lib/python3.10/site-packages/packaging/_parser.py': 'packaging._parser', '/usr/local/lib/python3.10/site-packages/packaging/markers.py': 'packaging.markers', '/usr/local/lib/python3.10/http/__init__.py': 'http', '/usr/local/lib/python3.10/email/feedparser.py': 'email.feedparser', '/usr/local/lib/python3.10/email/parser.py': 'email.parser', '/usr/local/lib/python3.10/http/client.py': 'http.client', '/usr/local/lib/python3.10/urllib/response.py': 'urllib.response', '/usr/local/lib/python3.10/urllib/error.py': 'urllib.error', '/usr/local/lib/python3.10/urllib/request.py': 'urllib.request', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/backports/__init__.py': 'backports', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/__init__.py': 'backports.tarfile.compat', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/compat/py38.py': 'backports.tarfile.compat.py38', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/backports/tarfile/__init__.py': 'backports.tarfile', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/jaraco/context.py': 'jaraco.context', '/usr/local/lib/python3.10/site-packages/setuptools/_vendor/jaraco/text/__init__.py': 'jaraco.text', '/usr/local/lib/python3.10/site-packages/setuptools/_importlib.py': 'setuptools._importlib', '/usr/local/lib/python3.10/site-packages/setuptools/_itertools.py': 'setuptools._itertools', '/usr/local/lib/python3.10/site-packages/setuptools/errors.py': 'setuptools.errors', '/usr/local/lib/python3.10/site-packages/setuptools/_entry_points.py': 'setuptools._entry_points', '/usr/local/lib/python3.10/site-packages/packaging/requirements.py': 'packaging.requirements', '/usr/local/lib/python3.10/site-packages/setuptools/_reqs.py': 'setuptools._reqs', '/usr/local/lib/python3.10/site-packages/setuptools/warnings.py': 'setuptools.warnings', '/usr/local/lib/python3.10/site-packages/setuptools/_static.py': 'setuptools._static', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/command/__init__.py': 'distutils.command', '/usr/local/lib/python3.10/site-packages/setuptools/_distutils/command/bdist.py': 'distutils.command.bdist', '/usr/local/lib/python3.10/site-packages/setuptools/command/__init__.py': 'setuptools.command', '/usr/local/lib/python3.10/site-packages/packaging/licenses/_spdx.py': 'packaging.licenses._spdx', '/usr/local/lib/python3.10/site-packages/packaging/licenses/__init__.py': 'packaging.licenses', '/usr/local/lib/python3.10/site-packages/setuptools/_normalization.py': 'setuptools._normalization', '/usr/local/lib/python3.10/configparser.py': 'configparser', '/usr/local/lib/python3.10/site-packages/setuptools/config/expand.py': 'setuptools.config.expand', '/usr/local/lib/python3.10/site-packages/setuptools/config/setupcfg.py': 'setuptools.config.setupcfg', '/usr/local/lib/python3.10/site-packages/setuptools/config/__init__.py': 'setuptools.config', '/usr/local/lib/python3.10/email/_header_value_parser.py': 'email._header_value_parser', '/usr/local/lib/python3.10/email/headerregistry.py': 'email.headerregistry', '/usr/local/lib/python3.10/site-packages/setuptools/extension.py': 'setuptools.extension', '/usr/local/lib/python3.10/site-packages/setuptools/config/_apply_pyprojecttoml.py': 'setuptools.config._apply_pyprojecttoml', '/usr/local/lib/python3.10/site-packages/setuptools/config/pyprojecttoml.py': 'setuptools.config.pyprojecttoml', '/usr/local/lib/python3.10/site-packages/setuptools/dist.py': 'setuptools.dist', '/usr/local/lib/python3.10/site-packages/setuptools/version.py': 'setuptools.version', '/usr/local/lib/python3.10/site-packages/setuptools/_core_metadata.py': 'setuptools._core_metadata', '/usr/local/lib/python3.10/site-packages/setuptools/__init__.py': 'setuptools', '/workspace/pynguin/analyses/constants.py': 'pynguin.analyses.constants', '/workspace/pynguin/llm/astscoping.py': 'pynguin.llm.astscoping', '/workspace/pynguin/utils/mutation_utils.py': 'pynguin.utils.mutation_utils', '/workspace/pynguin/testcase/statement.py': 'pynguin.testcase.statement', '/workspace/pynguin/utils/opcodes.py': 'pynguin.utils.opcodes', '/workspace/pynguin/instrumentation/__init__.py': 'pynguin.instrumentation', '/workspace/pynguin/analyses/controlflow.py': 'pynguin.analyses.controlflow', '/workspace/pynguin/instrumentation/instrumentation.py': 'pynguin.instrumentation.instrumentation', '/workspace/pynguin/testcase/statement_to_ast.py': 'pynguin.testcase.statement_to_ast', '/workspace/pynguin/utils/mirror.py': 'pynguin.utils.mirror', '/workspace/pynguin/testcase/execution.py': 'pynguin.testcase.execution', '/usr/local/lib/python3.10/fractions.py': 'fractions', '/usr/local/lib/python3.10/lib-dynload/_statistics.cpython-310-x86_64-linux-gnu.so': '_statistics', '/usr/local/lib/python3.10/statistics.py': 'statistics', '/workspace/pynguin/slicer/executionflowbuilder.py': 'pynguin.slicer.executionflowbuilder', '/workspace/pynguin/slicer/stack/__init__.py': 'pynguin.slicer.stack', '/workspace/pynguin/slicer/stack/stackeffect.py': 'pynguin.slicer.stack.stackeffect', '/workspace/pynguin/slicer/stack/stacksimulation.py': 'pynguin.slicer.stack.stacksimulation', '/workspace/pynguin/slicer/dynamicslicer.py': 'pynguin.slicer.dynamicslicer', '/workspace/pynguin/ga/computations.py': 'pynguin.ga.computations', '/workspace/pynguin/ga/chromosome.py': 'pynguin.ga.chromosome', '/workspace/pynguin/utils/statistics/statisticsbackend.py': 'pynguin.utils.statistics.statisticsbackend', '/workspace/pynguin/utils/statistics/outputvariablefactory.py': 'pynguin.utils.statistics.outputvariablefactory', '/workspace/pynguin/utils/statistics/stats.py': 'pynguin.utils.statistics.stats', '/workspace/pynguin/instrumentation/machinery.py': 'pynguin.instrumentation.machinery', '/workspace/pynguin/testcase/testcase.py': 'pynguin.testcase.testcase', '/workspace/pynguin/assertion/assertiontraceobserver.py': 'pynguin.assertion.assertiontraceobserver', '/workspace/pynguin/assertion/mutation_analysis/__init__.py': 'pynguin.assertion.mutation_analysis', '/workspace/pynguin/assertion/mutation_analysis/transformer.py': 'pynguin.assertion.mutation_analysis.transformer', '/workspace/pynguin/assertion/mutation_analysis/controller.py': 'pynguin.assertion.mutation_analysis.controller', '/workspace/pynguin/assertion/mutation_analysis/strategies.py': 'pynguin.assertion.mutation_analysis.strategies', '/workspace/pynguin/assertion/mutation_analysis/mutators.py': 'pynguin.assertion.mutation_analysis.mutators', '/workspace/pynguin/assertion/assertiongenerator.py': 'pynguin.assertion.assertiongenerator', '/workspace/pynguin/assertion/mutation_analysis/operators/base.py': 'pynguin.assertion.mutation_analysis.operators.base', '/workspace/pynguin/assertion/mutation_analysis/operators/arithmetic.py': 'pynguin.assertion.mutation_analysis.operators.arithmetic', '/workspace/pynguin/assertion/mutation_analysis/operators/decorator.py': 'pynguin.assertion.mutation_analysis.operators.decorator', '/workspace/pynguin/assertion/mutation_analysis/operators/exception.py': 'pynguin.assertion.mutation_analysis.operators.exception', '/workspace/pynguin/assertion/mutation_analysis/operators/inheritance.py': 'pynguin.assertion.mutation_analysis.operators.inheritance', '/workspace/pynguin/assertion/mutation_analysis/operators/logical.py': 'pynguin.assertion.mutation_analysis.operators.logical', '/workspace/pynguin/assertion/mutation_analysis/operators/loop.py': 'pynguin.assertion.mutation_analysis.operators.loop', '/workspace/pynguin/assertion/mutation_analysis/operators/misc.py': 'pynguin.assertion.mutation_analysis.operators.misc', '/workspace/pynguin/assertion/mutation_analysis/operators/__init__.py': 'pynguin.assertion.mutation_analysis.operators', '/workspace/pynguin/ga/algorithms/__init__.py': 'pynguin.ga.algorithms', '/workspace/pynguin/ga/testcasechromosome.py': 'pynguin.ga.testcasechromosome', '/workspace/pynguin/ga/algorithms/archive.py': 'pynguin.ga.algorithms.archive', '/workspace/pynguin/ga/searchobserver.py': 'pynguin.ga.searchobserver', '/workspace/pynguin/utils/statistics/statisticsobserver.py': 'pynguin.utils.statistics.statisticsobserver', '/workspace/pynguin/testcase/defaulttestcase.py': 'pynguin.testcase.defaulttestcase', '/workspace/pynguin/setup/__init__.py': 'pynguin.setup', '/workspace/pynguin/setup/testcluster.py': 'pynguin.setup.testcluster', '/workspace/pynguin/llm/stmtdeserializer.py': 'pynguin.llm.stmtdeserializer', '/workspace/pynguin/llm/ast_to_testcase.py': 'pynguin.llm.ast_to_testcase', '/workspace/pynguin/analyses/seeding.py': 'pynguin.analyses.seeding', '/workspace/pynguin/export/__init__.py': 'pynguin.export', '/workspace/pynguin/testcase/testcasevisitor.py': 'pynguin.testcase.testcasevisitor', '/workspace/pynguin/testcase/testcase_to_ast.py': 'pynguin.testcase.testcase_to_ast', '/workspace/pynguin/export/abstractexporter.py': 'pynguin.export.abstractexporter', '/workspace/pynguin/export/pytestexporter.py': 'pynguin.export.pytestexporter', '/workspace/pynguin/ga/testsuitechromosome.py': 'pynguin.ga.testsuitechromosome', '/workspace/pynguin/ga/algorithms/generationalgorithm.py': 'pynguin.ga.algorithms.generationalgorithm', '/workspace/pynguin/ga/operators/__init__.py': 'pynguin.ga.operators', '/workspace/pynguin/ga/operators/comparator.py': 'pynguin.ga.operators.comparator', '/workspace/pynguin/ga/algorithms/abstractmosaalgorithm.py': 'pynguin.ga.algorithms.abstractmosaalgorithm', '/workspace/pynguin/ga/operators/ranking.py': 'pynguin.ga.operators.ranking', '/usr/local/lib/python3.10/site-packages/langchain_core/_import_utils.py': 'langchain_core._import_utils', '/usr/local/lib/python3.10/site-packages/langchain_core/_api/__init__.py': 'langchain_core._api', '/usr/local/lib/python3.10/site-packages/langchain_core/_api/internal.py': 'langchain_core._api.internal', '/usr/local/lib/python3.10/site-packages/langchain_core/_api/beta_decorator.py': 'langchain_core._api.beta_decorator', '/usr/local/lib/python3.10/site-packages/pydantic_core/_pydantic_core.cpython-310-x86_64-linux-gnu.so': 'pydantic_core._pydantic_core', '/usr/local/lib/python3.10/site-packages/pydantic_core/core_schema.py': 'pydantic_core.core_schema', '/usr/local/lib/python3.10/site-packages/pydantic_core/__init__.py': 'pydantic_core', '/usr/local/lib/python3.10/site-packages/pydantic/version.py': 'pydantic.version', '/usr/local/lib/python3.10/site-packages/pydantic/warnings.py': 'pydantic.warnings', '/usr/local/lib/python3.10/site-packages/pydantic/_migration.py': 'pydantic._migration', '/usr/local/lib/python3.10/site-packages/typing_inspection/__init__.py': 'typing_inspection', '/usr/local/lib/python3.10/site-packages/typing_inspection/typing_objects.py': 'typing_inspection.typing_objects', '/usr/local/lib/python3.10/site-packages/typing_inspection/introspection.py': 'typing_inspection.introspection', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/__init__.py': 'pydantic._internal', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_namespace_utils.py': 'pydantic._internal._namespace_utils', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_typing_extra.py': 'pydantic._internal._typing_extra', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_repr.py': 'pydantic._internal._repr', '/usr/local/lib/python3.10/site-packages/pydantic/errors.py': 'pydantic.errors', '/usr/local/lib/python3.10/site-packages/pydantic/__init__.py': 'pydantic', '/usr/local/lib/python3.10/site-packages/annotated_types/__init__.py': 'annotated_types', '/usr/local/lib/python3.10/_sysconfigdata__linux_x86_64-linux-gnu.py': '_sysconfigdata__linux_x86_64-linux-gnu', '/usr/local/lib/python3.10/zoneinfo/_tzpath.py': 'zoneinfo._tzpath', '/usr/local/lib/python3.10/zoneinfo/_common.py': 'zoneinfo._common', '/usr/local/lib/python3.10/lib-dynload/_zoneinfo.cpython-310-x86_64-linux-gnu.so': '_zoneinfo', '/usr/local/lib/python3.10/zoneinfo/__init__.py': 'zoneinfo', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_import_utils.py': 'pydantic._internal._import_utils', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_validators.py': 'pydantic._internal._validators', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_internal_dataclass.py': 'pydantic._internal._internal_dataclass', '/usr/local/lib/python3.10/site-packages/pydantic/aliases.py': 'pydantic.aliases', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_core_utils.py': 'pydantic._internal._core_utils', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_forward_ref.py': 'pydantic._internal._forward_ref', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_utils.py': 'pydantic._internal._utils', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_generics.py': 'pydantic._internal._generics', '/usr/local/lib/python3.10/site-packages/pydantic/config.py': 'pydantic.config', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_config.py': 'pydantic._internal._config', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_docs_extraction.py': 'pydantic._internal._docs_extraction', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_fields.py': 'pydantic._internal._fields', '/usr/local/lib/python3.10/site-packages/pydantic/annotated_handlers.py': 'pydantic.annotated_handlers', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_core_metadata.py': 'pydantic._internal._core_metadata', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_decorators.py': 'pydantic._internal._decorators', '/usr/local/lib/python3.10/site-packages/pydantic/plugin/__init__.py': 'pydantic.plugin', '/usr/local/lib/python3.10/site-packages/pydantic/plugin/_schema_validator.py': 'pydantic.plugin._schema_validator', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_mock_val_ser.py': 'pydantic._internal._mock_val_ser', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_schema_generation_shared.py': 'pydantic._internal._schema_generation_shared', '/usr/local/lib/python3.10/site-packages/pydantic/json_schema.py': 'pydantic.json_schema', '/usr/local/lib/python3.10/site-packages/pydantic/types.py': 'pydantic.types', '/usr/local/lib/python3.10/site-packages/pydantic/fields.py': 'pydantic.fields', '/usr/local/lib/python3.10/site-packages/langchain_core/_api/deprecation.py': 'langchain_core._api.deprecation', '/usr/local/lib/python3.10/site-packages/langchain_core/version.py': 'langchain_core.version', '/usr/local/lib/python3.10/site-packages/langchain_core/__init__.py': 'langchain_core', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/__init__.py': 'langchain_core.utils', '/usr/local/lib/python3.10/site-packages/urllib3/exceptions.py': 'urllib3.exceptions', '/usr/local/lib/python3.10/site-packages/urllib3/util/timeout.py': 'urllib3.util.timeout', '/usr/local/lib/python3.10/site-packages/urllib3/util/connection.py': 'urllib3.util.connection', '/usr/local/lib/python3.10/site-packages/urllib3/util/util.py': 'urllib3.util.util', '/usr/local/lib/python3.10/site-packages/urllib3/util/request.py': 'urllib3.util.request', '/usr/local/lib/python3.10/site-packages/urllib3/util/response.py': 'urllib3.util.response', '/usr/local/lib/python3.10/site-packages/urllib3/util/retry.py': 'urllib3.util.retry', '/usr/local/lib/python3.10/hmac.py': 'hmac', '/usr/local/lib/python3.10/site-packages/urllib3/util/url.py': 'urllib3.util.url', '/usr/local/lib/python3.10/site-packages/urllib3/util/ssltransport.py': 'urllib3.util.ssltransport', '/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_.py': 'urllib3.util.ssl_', '/usr/local/lib/python3.10/site-packages/urllib3/util/wait.py': 'urllib3.util.wait', '/usr/local/lib/python3.10/site-packages/urllib3/util/__init__.py': 'urllib3.util', '/usr/local/lib/python3.10/site-packages/urllib3/_base_connection.py': 'urllib3._base_connection', '/usr/local/lib/python3.10/site-packages/urllib3/_collections.py': 'urllib3._collections', '/usr/local/lib/python3.10/site-packages/urllib3/_version.py': 'urllib3._version', '/usr/local/lib/python3.10/mimetypes.py': 'mimetypes', '/usr/local/lib/python3.10/site-packages/urllib3/fields.py': 'urllib3.fields', '/usr/local/lib/python3.10/site-packages/urllib3/filepost.py': 'urllib3.filepost', '/usr/local/lib/python3.10/site-packages/urllib3/http2/__init__.py': 'urllib3.http2', '/usr/local/lib/python3.10/site-packages/urllib3/http2/probe.py': 'urllib3.http2.probe', '/usr/local/lib/python3.10/site-packages/urllib3/util/ssl_match_hostname.py': 'urllib3.util.ssl_match_hostname', '/usr/local/lib/python3.10/site-packages/urllib3/connection.py': 'urllib3.connection', '/usr/local/lib/python3.10/site-packages/urllib3/response.py': 'urllib3.response', '/usr/local/lib/python3.10/site-packages/urllib3/_request_methods.py': 'urllib3._request_methods', '/usr/local/lib/python3.10/site-packages/urllib3/util/proxy.py': 'urllib3.util.proxy', '/usr/local/lib/python3.10/site-packages/urllib3/connectionpool.py': 'urllib3.connectionpool', '/usr/local/lib/python3.10/site-packages/urllib3/poolmanager.py': 'urllib3.poolmanager', '/usr/local/lib/python3.10/site-packages/urllib3/__init__.py': 'urllib3', '/usr/local/lib/python3.10/site-packages/ada92cb5d92a588d1b93__mypyc.cpython-310-x86_64-linux-gnu.so': 'ada92cb5d92a588d1b93__mypyc', '/usr/local/lib/python3.10/site-packages/charset_normalizer/constant.py': 'charset_normalizer.constant', '/usr/local/lib/python3.10/site-packages/charset_normalizer/md.cpython-310-x86_64-linux-gnu.so': 'charset_normalizer.md', '/usr/local/lib/python3.10/site-packages/charset_normalizer/utils.py': 'charset_normalizer.utils', '/usr/local/lib/python3.10/site-packages/charset_normalizer/models.py': 'charset_normalizer.models', '/usr/local/lib/python3.10/site-packages/charset_normalizer/cd.cpython-310-x86_64-linux-gnu.so': 'charset_normalizer.cd', '/usr/local/lib/python3.10/lib-dynload/_codecs_cn.cpython-310-x86_64-linux-gnu.so': '_codecs_cn', '/usr/local/lib/python3.10/lib-dynload/_multibytecodec.cpython-310-x86_64-linux-gnu.so': '_multibytecodec', '/usr/local/lib/python3.10/lib-dynload/_codecs_hk.cpython-310-x86_64-linux-gnu.so': '_codecs_hk', '/usr/local/lib/python3.10/lib-dynload/_codecs_iso2022.cpython-310-x86_64-linux-gnu.so': '_codecs_iso2022', '/usr/local/lib/python3.10/lib-dynload/_codecs_jp.cpython-310-x86_64-linux-gnu.so': '_codecs_jp', '/usr/local/lib/python3.10/lib-dynload/_codecs_kr.cpython-310-x86_64-linux-gnu.so': '_codecs_kr', '/usr/local/lib/python3.10/lib-dynload/_codecs_tw.cpython-310-x86_64-linux-gnu.so': '_codecs_tw', '/usr/local/lib/python3.10/site-packages/charset_normalizer/api.py': 'charset_normalizer.api', '/usr/local/lib/python3.10/site-packages/charset_normalizer/legacy.py': 'charset_normalizer.legacy', '/usr/local/lib/python3.10/site-packages/charset_normalizer/version.py': 'charset_normalizer.version', '/usr/local/lib/python3.10/site-packages/charset_normalizer/__init__.py': 'charset_normalizer', '/usr/local/lib/python3.10/http/cookiejar.py': 'http.cookiejar', '/usr/local/lib/python3.10/http/cookies.py': 'http.cookies', '/usr/local/lib/python3.10/site-packages/requests/compat.py': 'requests.compat', '/usr/local/lib/python3.10/site-packages/requests/exceptions.py': 'requests.exceptions', '/usr/local/lib/python3.10/site-packages/idna/idnadata.py': 'idna.idnadata', '/usr/local/lib/python3.10/site-packages/idna/intranges.py': 'idna.intranges', '/usr/local/lib/python3.10/site-packages/idna/core.py': 'idna.core', '/usr/local/lib/python3.10/site-packages/idna/package_data.py': 'idna.package_data', '/usr/local/lib/python3.10/site-packages/idna/__init__.py': 'idna', '/usr/local/lib/python3.10/site-packages/requests/packages.py': 'requests.packages', '/usr/local/lib/python3.10/site-packages/certifi/core.py': 'certifi.core', '/usr/local/lib/python3.10/site-packages/certifi/__init__.py': 'certifi', '/usr/local/lib/python3.10/site-packages/requests/certs.py': 'requests.certs', '/usr/local/lib/python3.10/site-packages/requests/__version__.py': 'requests.__version__', '/usr/local/lib/python3.10/site-packages/requests/_internal_utils.py': 'requests._internal_utils', '/usr/local/lib/python3.10/site-packages/requests/_types.py': 'requests._types', '/usr/local/lib/python3.10/site-packages/requests/cookies.py': 'requests.cookies', '/usr/local/lib/python3.10/site-packages/requests/structures.py': 'requests.structures', '/usr/local/lib/python3.10/site-packages/requests/utils.py': 'requests.utils', '/usr/local/lib/python3.10/site-packages/requests/auth.py': 'requests.auth', '/usr/local/lib/python3.10/stringprep.py': 'stringprep', '/usr/local/lib/python3.10/encodings/idna.py': 'encodings.idna', '/usr/local/lib/python3.10/site-packages/requests/hooks.py': 'requests.hooks', '/usr/local/lib/python3.10/site-packages/requests/status_codes.py': 'requests.status_codes', '/usr/local/lib/python3.10/site-packages/requests/models.py': 'requests.models', '/usr/local/lib/python3.10/site-packages/urllib3/contrib/__init__.py': 'urllib3.contrib', '/usr/local/lib/python3.10/site-packages/requests/adapters.py': 'requests.adapters', '/usr/local/lib/python3.10/site-packages/requests/sessions.py': 'requests.sessions', '/usr/local/lib/python3.10/site-packages/requests/api.py': 'requests.api', '/usr/local/lib/python3.10/site-packages/requests/__init__.py': 'requests', '/usr/local/lib/python3.10/site-packages/pydantic/functional_validators.py': 'pydantic.functional_validators', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_discriminated_union.py': 'pydantic._internal._discriminated_union', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_known_annotated_metadata.py': 'pydantic._internal._known_annotated_metadata', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_schema_gather.py': 'pydantic._internal._schema_gather', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_generate_schema.py': 'pydantic._internal._generate_schema', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_signature.py': 'pydantic._internal._signature', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_model_construction.py': 'pydantic._internal._model_construction', '/usr/local/lib/python3.10/site-packages/pydantic/main.py': 'pydantic.main', '/usr/local/lib/python3.10/site-packages/pydantic/plugin/_loader.py': 'pydantic.plugin._loader', '/usr/local/lib/python3.10/site-packages/pydantic/root_model.py': 'pydantic.root_model', '/usr/local/lib/python3.10/site-packages/pydantic/deprecated/__init__.py': 'pydantic.deprecated', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_decorators_v1.py': 'pydantic._internal._decorators_v1', '/usr/local/lib/python3.10/site-packages/pydantic/deprecated/class_validators.py': 'pydantic.deprecated.class_validators', '/usr/local/lib/python3.10/site-packages/pydantic/v1/typing.py': 'pydantic.v1.typing', '/usr/local/lib/python3.10/site-packages/pydantic/v1/errors.py': 'pydantic.v1.errors', '/usr/local/lib/python3.10/site-packages/pydantic/v1/version.py': 'pydantic.v1.version', '/usr/local/lib/python3.10/site-packages/pydantic/v1/utils.py': 'pydantic.v1.utils', '/usr/local/lib/python3.10/site-packages/pydantic/v1/class_validators.py': 'pydantic.v1.class_validators', '/usr/local/lib/python3.10/site-packages/pydantic/v1/config.py': 'pydantic.v1.config', '/usr/local/lib/python3.10/colorsys.py': 'colorsys', '/usr/local/lib/python3.10/site-packages/pydantic/v1/color.py': 'pydantic.v1.color', '/usr/local/lib/python3.10/site-packages/pydantic/v1/datetime_parse.py': 'pydantic.v1.datetime_parse', '/usr/local/lib/python3.10/site-packages/pydantic/v1/validators.py': 'pydantic.v1.validators', '/usr/local/lib/python3.10/site-packages/pydantic/v1/networks.py': 'pydantic.v1.networks', '/usr/local/lib/python3.10/site-packages/pydantic/v1/types.py': 'pydantic.v1.types', '/usr/local/lib/python3.10/site-packages/pydantic/v1/json.py': 'pydantic.v1.json', '/usr/local/lib/python3.10/site-packages/pydantic/v1/error_wrappers.py': 'pydantic.v1.error_wrappers', '/usr/local/lib/python3.10/site-packages/pydantic/v1/fields.py': 'pydantic.v1.fields', '/usr/local/lib/python3.10/site-packages/pydantic/v1/parse.py': 'pydantic.v1.parse', '/usr/local/lib/python3.10/site-packages/pydantic/v1/schema.py': 'pydantic.v1.schema', '/usr/local/lib/python3.10/site-packages/pydantic/v1/main.py': 'pydantic.v1.main', '/usr/local/lib/python3.10/site-packages/pydantic/v1/dataclasses.py': 'pydantic.v1.dataclasses', '/usr/local/lib/python3.10/site-packages/pydantic/v1/annotated_types.py': 'pydantic.v1.annotated_types', '/usr/local/lib/python3.10/site-packages/pydantic/v1/decorator.py': 'pydantic.v1.decorator', '/usr/local/lib/python3.10/site-packages/pydantic/v1/env_settings.py': 'pydantic.v1.env_settings', '/usr/local/lib/python3.10/site-packages/pydantic/v1/tools.py': 'pydantic.v1.tools', '/usr/local/lib/python3.10/site-packages/pydantic/v1/__init__.py': 'pydantic.v1', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/pydantic.py': 'langchain_core.utils.pydantic', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/utils.py': 'langchain_core.utils.utils', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/__init__.py': 'langchain_core.messages', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/content.py': 'langchain_core.messages.content', '/usr/local/lib/python3.10/site-packages/langchain_core/load/serializable.py': 'langchain_core.load.serializable', '/usr/local/lib/python3.10/site-packages/langchain_core/load/_validation.py': 'langchain_core.load._validation', '/usr/local/lib/python3.10/site-packages/langchain_core/load/mapping.py': 'langchain_core.load.mapping', '/usr/local/lib/python3.10/site-packages/langchain_core/load/load.py': 'langchain_core.load.load', '/usr/local/lib/python3.10/site-packages/langchain_core/load/__init__.py': 'langchain_core.load', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/input.py': 'langchain_core.utils.input', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/_merge.py': 'langchain_core.utils._merge', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/interactive_env.py': 'langchain_core.utils.interactive_env', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/base.py': 'langchain_core.messages.base', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/tool.py': 'langchain_core.messages.tool', '/usr/local/lib/python3.10/site-packages/langchain_core/exceptions.py': 'langchain_core.exceptions', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/json.py': 'langchain_core.utils.json', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/usage.py': 'langchain_core.utils.usage', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/ai.py': 'langchain_core.messages.ai', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/__init__.py': 'langchain_core.prompts', '/usr/local/lib/python3.10/xml/sax/handler.py': 'xml.sax.handler', '/usr/local/lib/python3.10/xml/sax/_exceptions.py': 'xml.sax._exceptions', '/usr/local/lib/python3.10/xml/sax/xmlreader.py': 'xml.sax.xmlreader', '/usr/local/lib/python3.10/xml/sax/__init__.py': 'xml.sax', '/usr/local/lib/python3.10/xml/sax/saxutils.py': 'xml.sax.saxutils', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/anthropic.py': 'langchain_core.messages.block_translators.anthropic', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/bedrock.py': 'langchain_core.messages.block_translators.bedrock', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/bedrock_converse.py': 'langchain_core.messages.block_translators.bedrock_converse', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/google_genai.py': 'langchain_core.messages.block_translators.google_genai', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/google_vertexai.py': 'langchain_core.messages.block_translators.google_vertexai', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/groq.py': 'langchain_core.messages.block_translators.groq', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/_utils.py': 'langchain_core.language_models._utils', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/__init__.py': 'langchain_core.language_models', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/openai.py': 'langchain_core.messages.block_translators.openai', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/__init__.py': 'langchain_core.messages.block_translators', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/chat.py': 'langchain_core.messages.chat', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/function.py': 'langchain_core.messages.function', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/human.py': 'langchain_core.messages.human', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/modifier.py': 'langchain_core.messages.modifier', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/system.py': 'langchain_core.messages.system', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/utils.py': 'langchain_core.messages.utils', '/usr/local/lib/python3.10/site-packages/pydantic/_internal/_serializers.py': 'pydantic._internal._serializers', '/usr/local/lib/python3.10/site-packages/langchain_core/prompt_values.py': 'langchain_core.prompt_values', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/__init__.py': 'langchain_core.outputs', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/generation.py': 'langchain_core.outputs.generation', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/chat_generation.py': 'langchain_core.outputs.chat_generation', '/usr/local/lib/python3.10/site-packages/langchain_core/load/dump.py': 'langchain_core.load.dump', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/__init__.py': 'langchain_core.output_parsers', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/__init__.py': 'langchain_core.runnables', '/usr/local/lib/python3.10/concurrent/futures/thread.py': 'concurrent.futures.thread', '/usr/local/lib/python3.10/site-packages/langchain_core/callbacks/__init__.py': 'langchain_core.callbacks', '/usr/local/lib/python3.10/site-packages/langchain_core/callbacks/base.py': 'langchain_core.callbacks.base', '/usr/local/lib/python3.10/site-packages/langchain_core/callbacks/stdout.py': 'langchain_core.callbacks.stdout', '/usr/local/lib/python3.10/site-packages/langchain_core/globals.py': 'langchain_core.globals', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/env.py': 'langchain_core.utils.env', '/usr/local/lib/python3.10/site-packages/uuid_utils/_uuid_utils.cpython-310-x86_64-linux-gnu.so': 'uuid_utils._uuid_utils', '/usr/local/lib/python3.10/site-packages/uuid_utils/__init__.py': 'uuid_utils', '/usr/local/lib/python3.10/site-packages/uuid_utils/compat/__init__.py': 'uuid_utils.compat', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/uuid.py': 'langchain_core.utils.uuid', '/usr/local/lib/python3.10/site-packages/langchain_core/callbacks/manager.py': 'langchain_core.callbacks.manager', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/utils.py': 'langchain_core.runnables.utils', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/config.py': 'langchain_core.runnables.config', '/usr/local/lib/python3.10/site-packages/langchain_core/caches.py': 'langchain_core.caches', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/__init__.py': 'langchain_core.tracers', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/_streaming.py': 'langchain_core.tracers._streaming', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/run_info.py': 'langchain_core.outputs.run_info', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/llm_result.py': 'langchain_core.outputs.llm_result', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/schema.py': 'langchain_core.runnables.schema', '/usr/local/lib/python3.10/site-packages/jsonpointer.py': 'jsonpointer', '/usr/local/lib/python3.10/site-packages/jsonpatch.py': 'jsonpatch', '/usr/local/lib/python3.10/site-packages/langsmith/__init__.py': 'langsmith', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/__init__.py': 'langsmith._internal', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_context.py': 'langsmith._internal._context', '/usr/local/lib/python3.10/site-packages/langsmith/schemas.py': 'langsmith.schemas', '/usr/local/lib/python3.10/site-packages/httpx/__version__.py': 'httpx.__version__', '/usr/local/lib/python3.10/site-packages/httpx/_exceptions.py': 'httpx._exceptions', '/usr/local/lib/python3.10/site-packages/httpx/_types.py': 'httpx._types', '/usr/local/lib/python3.10/site-packages/httpx/_utils.py': 'httpx._utils', '/usr/local/lib/python3.10/site-packages/httpx/_multipart.py': 'httpx._multipart', '/usr/local/lib/python3.10/site-packages/httpx/_content.py': 'httpx._content', '/usr/local/lib/python3.10/site-packages/zstandard/backend_c.cpython-310-x86_64-linux-gnu.so': 'zstandard.backend_c', '/usr/local/lib/python3.10/site-packages/zstandard/__init__.py': 'zstandard', '/usr/local/lib/python3.10/site-packages/httpx/_decoders.py': 'httpx._decoders', '/usr/local/lib/python3.10/site-packages/httpx/_status_codes.py': 'httpx._status_codes', '/usr/local/lib/python3.10/site-packages/httpx/_urlparse.py': 'httpx._urlparse', '/usr/local/lib/python3.10/site-packages/httpx/_urls.py': 'httpx._urls', '/usr/local/lib/python3.10/site-packages/httpx/_models.py': 'httpx._models', '/usr/local/lib/python3.10/site-packages/httpx/_auth.py': 'httpx._auth', '/usr/local/lib/python3.10/site-packages/httpx/_config.py': 'httpx._config', '/usr/local/lib/python3.10/site-packages/httpx/_transports/base.py': 'httpx._transports.base', '/usr/local/lib/python3.10/site-packages/httpx/_transports/asgi.py': 'httpx._transports.asgi', '/usr/local/lib/python3.10/site-packages/httpx/_transports/default.py': 'httpx._transports.default', '/usr/local/lib/python3.10/site-packages/httpx/_transports/mock.py': 'httpx._transports.mock', '/usr/local/lib/python3.10/site-packages/httpx/_transports/wsgi.py': 'httpx._transports.wsgi', '/usr/local/lib/python3.10/site-packages/httpx/_transports/__init__.py': 'httpx._transports', '/usr/local/lib/python3.10/site-packages/httpx/_client.py': 'httpx._client', '/usr/local/lib/python3.10/site-packages/httpx/_api.py': 'httpx._api', '/usr/local/lib/python3.10/site-packages/click/_compat.py': 'click._compat', '/usr/local/lib/python3.10/site-packages/click/globals.py': 'click.globals', '/usr/local/lib/python3.10/site-packages/click/utils.py': 'click.utils', '/usr/local/lib/python3.10/site-packages/click/exceptions.py': 'click.exceptions', '/usr/local/lib/python3.10/site-packages/click/types.py': 'click.types', '/usr/local/lib/python3.10/site-packages/click/_utils.py': 'click._utils', '/usr/local/lib/python3.10/site-packages/click/parser.py': 'click.parser', '/usr/local/lib/python3.10/site-packages/click/formatting.py': 'click.formatting', '/usr/local/lib/python3.10/site-packages/click/termui.py': 'click.termui', '/usr/local/lib/python3.10/site-packages/click/core.py': 'click.core', '/usr/local/lib/python3.10/site-packages/click/decorators.py': 'click.decorators', '/usr/local/lib/python3.10/site-packages/click/__init__.py': 'click', '/usr/local/lib/python3.10/site-packages/httpx/__init__.py': 'httpx', '/usr/local/lib/python3.10/site-packages/langsmith/utils.py': 'langsmith.utils', '/usr/local/lib/python3.10/site-packages/xxhash/_xxhash.cpython-310-x86_64-linux-gnu.so': 'xxhash._xxhash', '/usr/local/lib/python3.10/site-packages/xxhash/version.py': 'xxhash.version', '/usr/local/lib/python3.10/site-packages/xxhash/__init__.py': 'xxhash', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_uuid.py': 'langsmith._internal._uuid', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/_compat.py': 'requests_toolbelt._compat', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/adapters/ssl.py': 'requests_toolbelt.adapters.ssl', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/adapters/source.py': 'requests_toolbelt.adapters.source', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/adapters/__init__.py': 'requests_toolbelt.adapters', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/auth/__init__.py': 'requests_toolbelt.auth', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/auth/_digest_auth_compat.py': 'requests_toolbelt.auth._digest_auth_compat', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/auth/http_proxy_digest.py': 'requests_toolbelt.auth.http_proxy_digest', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/auth/guess.py': 'requests_toolbelt.auth.guess', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/multipart/encoder.py': 'requests_toolbelt.multipart.encoder', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/multipart/decoder.py': 'requests_toolbelt.multipart.decoder', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/multipart/__init__.py': 'requests_toolbelt.multipart', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/streaming_iterator.py': 'requests_toolbelt.streaming_iterator', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/utils/__init__.py': 'requests_toolbelt.utils', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/utils/user_agent.py': 'requests_toolbelt.utils.user_agent', '/usr/local/lib/python3.10/site-packages/requests_toolbelt/__init__.py': 'requests_toolbelt', '/usr/local/lib/python3.10/site-packages/langsmith/env/_git.py': 'langsmith.env._git', '/usr/local/lib/python3.10/site-packages/langsmith/env/_runtime_env.py': 'langsmith.env._runtime_env', '/usr/local/lib/python3.10/site-packages/langsmith/env/__init__.py': 'langsmith.env', '/usr/local/lib/python3.10/site-packages/langsmith/_runtime_overrides.py': 'langsmith._runtime_overrides', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_aiter.py': 'langsmith._internal._aiter', '/usr/local/lib/python3.10/site-packages/orjson/orjson.cpython-310-x86_64-linux-gnu.so': 'orjson.orjson', '/usr/local/lib/python3.10/site-packages/orjson/__init__.py': 'orjson', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_orjson.py': 'langsmith._internal._orjson', '/usr/local/lib/python3.10/secrets.py': 'secrets', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_oauth_refresh_lock.py': 'langsmith._internal._oauth_refresh_lock', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_profiles.py': 'langsmith._internal._profiles', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_constants.py': 'langsmith._internal._constants', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_backend_version.py': 'langsmith._internal._backend_version', '/usr/local/lib/python3.10/multiprocessing/process.py': 'multiprocessing.process', '/usr/local/lib/python3.10/multiprocessing/reduction.py': 'multiprocessing.reduction', '/usr/local/lib/python3.10/multiprocessing/context.py': 'multiprocessing.context', '/usr/local/lib/python3.10/multiprocessing/__init__.py': 'multiprocessing', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_compressed_traces.py': 'langsmith._internal._compressed_traces', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_multipart.py': 'langsmith._internal._multipart', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_serde.py': 'langsmith._internal._serde', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_operations.py': 'langsmith._internal._operations', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_background_thread.py': 'langsmith._internal._background_thread', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_beta_decorator.py': 'langsmith._internal._beta_decorator', '/usr/local/lib/python3.10/site-packages/langsmith/_internal/_hub.py': 'langsmith._internal._hub', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_types.py': 'langsmith._openapi_client._types', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_path.py': 'langsmith._openapi_client._utils._path', '/usr/local/lib/python3.10/site-packages/anyio/_core/__init__.py': 'anyio._core', '/usr/local/lib/python3.10/site-packages/anyio/_core/_contextmanagers.py': 'anyio._core._contextmanagers', '/usr/local/lib/python3.10/site-packages/anyio/_core/_exceptions.py': 'anyio._core._exceptions', '/usr/local/lib/python3.10/site-packages/sniffio/_version.py': 'sniffio._version', '/usr/local/lib/python3.10/site-packages/sniffio/_impl.py': 'sniffio._impl', '/usr/local/lib/python3.10/site-packages/sniffio/__init__.py': 'sniffio', '/usr/local/lib/python3.10/site-packages/anyio/_core/_eventloop.py': 'anyio._core._eventloop', '/usr/local/lib/python3.10/site-packages/anyio/to_thread.py': 'anyio.to_thread', '/usr/local/lib/python3.10/site-packages/anyio/abc/_eventloop.py': 'anyio.abc._eventloop', '/usr/local/lib/python3.10/site-packages/anyio/abc/_resources.py': 'anyio.abc._resources', '/usr/local/lib/python3.10/site-packages/anyio/_core/_typedattr.py': 'anyio._core._typedattr', '/usr/local/lib/python3.10/site-packages/anyio/abc/_tasks.py': 'anyio.abc._tasks', '/usr/local/lib/python3.10/site-packages/anyio/abc/_streams.py': 'anyio.abc._streams', '/usr/local/lib/python3.10/site-packages/anyio/abc/_sockets.py': 'anyio.abc._sockets', '/usr/local/lib/python3.10/site-packages/anyio/abc/_subprocesses.py': 'anyio.abc._subprocesses', '/usr/local/lib/python3.10/site-packages/anyio/abc/_testing.py': 'anyio.abc._testing', '/usr/local/lib/python3.10/site-packages/anyio/lowlevel.py': 'anyio.lowlevel', '/usr/local/lib/python3.10/site-packages/anyio/_core/_tasks.py': 'anyio._core._tasks', '/usr/local/lib/python3.10/site-packages/anyio/_core/_testing.py': 'anyio._core._testing', '/usr/local/lib/python3.10/site-packages/anyio/_core/_synchronization.py': 'anyio._core._synchronization', '/usr/local/lib/python3.10/site-packages/anyio/from_thread.py': 'anyio.from_thread', '/usr/local/lib/python3.10/site-packages/anyio/abc/__init__.py': 'anyio.abc', '/usr/local/lib/python3.10/site-packages/anyio/_core/_fileio.py': 'anyio._core._fileio', '/usr/local/lib/python3.10/site-packages/anyio/_core/_resources.py': 'anyio._core._resources', '/usr/local/lib/python3.10/site-packages/anyio/_core/_signals.py': 'anyio._core._signals', '/usr/local/lib/python3.10/site-packages/anyio/streams/__init__.py': 'anyio.streams', '/usr/local/lib/python3.10/site-packages/anyio/streams/stapled.py': 'anyio.streams.stapled', '/usr/local/lib/python3.10/site-packages/anyio/streams/tls.py': 'anyio.streams.tls', '/usr/local/lib/python3.10/site-packages/anyio/_core/_sockets.py': 'anyio._core._sockets', '/usr/local/lib/python3.10/site-packages/anyio/streams/memory.py': 'anyio.streams.memory', '/usr/local/lib/python3.10/site-packages/anyio/_core/_streams.py': 'anyio._core._streams', '/usr/local/lib/python3.10/site-packages/anyio/_core/_subprocesses.py': 'anyio._core._subprocesses', '/usr/local/lib/python3.10/site-packages/anyio/_core/_tempfile.py': 'anyio._core._tempfile', '/usr/local/lib/python3.10/site-packages/anyio/__init__.py': 'anyio', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_sync.py': 'langsmith._openapi_client._utils._sync', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_proxy.py': 'langsmith._openapi_client._utils._proxy', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_utils.py': 'langsmith._openapi_client._utils._utils', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_datetime_parse.py': 'langsmith._openapi_client._utils._datetime_parse', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_compat.py': 'langsmith._openapi_client._utils._compat', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_typing.py': 'langsmith._openapi_client._utils._typing', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_streams.py': 'langsmith._openapi_client._utils._streams', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_files.py': 'langsmith._openapi_client._files', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_transform.py': 'langsmith._openapi_client._utils._transform', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_reflection.py': 'langsmith._openapi_client._utils._reflection', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/__init__.py': 'langsmith._openapi_client._utils', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_compat.py': 'langsmith._openapi_client._compat', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_constants.py': 'langsmith._openapi_client._constants', '/usr/local/lib/python3.10/site-packages/pydantic/type_adapter.py': 'pydantic.type_adapter', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_models.py': 'langsmith._openapi_client._models', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_type.py': 'langsmith._openapi_client.types.run_type', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run.py': 'langsmith._openapi_client.types.run', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/issue.py': 'langsmith._openapi_client.types.issue', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/trace_aggregates.py': 'langsmith._openapi_client.types.trace_aggregates', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/trace.py': 'langsmith._openapi_client.types.trace', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread.py': 'langsmith._openapi_client.types.thread', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/data_type.py': 'langsmith._openapi_client.types.data_type', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread_stats.py': 'langsmith._openapi_client.types.thread_stats', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread_trace.py': 'langsmith._openapi_client.types.thread_trace', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_type_enum.py': 'langsmith._openapi_client.types.run_type_enum', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_llm_evaluator.py': 'langsmith._openapi_client.types.online_llm_evaluator', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_code_evaluator.py': 'langsmith._openapi_client.types.online_code_evaluator', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_type.py': 'langsmith._openapi_client.types.online_evaluator_type', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_spend_limit.py': 'langsmith._openapi_client.types.online_spend_limit', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_run_rule.py': 'langsmith._openapi_client.types.online_evaluator_run_rule', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator.py': 'langsmith._openapi_client.types.online_evaluator', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_select_field.py': 'langsmith._openapi_client.types.run_select_field', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_query_params.py': 'langsmith._openapi_client.types.run_query_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/sandbox_response.py': 'langsmith._openapi_client.types.sandbox_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/issue_list_params.py': 'langsmith._openapi_client.types.issue_list_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/snapshot_response.py': 'langsmith._openapi_client.types.snapshot_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/info_list_response.py': 'langsmith._openapi_client.types.info_list_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_get_url_params.py': 'langsmith._openapi_client.types.run_get_url_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/trace_query_params.py': 'langsmith._openapi_client.types.trace_query_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_query_v2_params.py': 'langsmith._openapi_client.types.run_query_v2_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_retrieve_params.py': 'langsmith._openapi_client.types.run_retrieve_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread_query_params.py': 'langsmith._openapi_client.types.thread_query_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread_stats_params.py': 'langsmith._openapi_client.types.thread_stats_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_get_url_response.py': 'langsmith._openapi_client.types.run_get_url_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/service_url_response.py': 'langsmith._openapi_client.types.service_url_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/sandbox_list_response.py': 'langsmith._openapi_client.types.sandbox_list_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/run_retrieve_v2_params.py': 'langsmith._openapi_client.types.run_retrieve_v2_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/snapshot_list_response.py': 'langsmith._openapi_client.types.snapshot_list_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/sort_by_dataset_column.py': 'langsmith._openapi_client.types.sort_by_dataset_column', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/trace_list_runs_params.py': 'langsmith._openapi_client.types.trace_list_runs_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/sandbox_status_response.py': 'langsmith._openapi_client.types.sandbox_status_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/trace_list_runs_response.py': 'langsmith._openapi_client.types.trace_list_runs_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/thread_list_traces_params.py': 'langsmith._openapi_client.types.thread_list_traces_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_spend_day.py': 'langsmith._openapi_client.types.online_evaluator_spend_day', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_list_params.py': 'langsmith._openapi_client.types.online_evaluator_list_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_spend_group.py': 'langsmith._openapi_client.types.online_evaluator_spend_group', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_spend_params.py': 'langsmith._openapi_client.types.online_evaluator_spend_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/create_online_llm_evaluator_request_param.py': 'langsmith._openapi_client.types.create_online_llm_evaluator_request_param', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/create_online_code_evaluator_request_param.py': 'langsmith._openapi_client.types.create_online_code_evaluator_request_param', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_create_params.py': 'langsmith._openapi_client.types.online_evaluator_create_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_delete_params.py': 'langsmith._openapi_client.types.online_evaluator_delete_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/update_online_llm_evaluator_request_param.py': 'langsmith._openapi_client.types.update_online_llm_evaluator_request_param', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/update_online_code_evaluator_request_param.py': 'langsmith._openapi_client.types.update_online_code_evaluator_request_param', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_update_params.py': 'langsmith._openapi_client.types.online_evaluator_update_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/bulk_delete_evaluator_failed_item.py': 'langsmith._openapi_client.types.bulk_delete_evaluator_failed_item', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/bulk_delete_evaluators_response.py': 'langsmith._openapi_client.types.bulk_delete_evaluators_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/create_online_evaluator_response.py': 'langsmith._openapi_client.types.create_online_evaluator_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/update_online_evaluator_response.py': 'langsmith._openapi_client.types.update_online_evaluator_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/runs_filter_data_source_type_enum.py': 'langsmith._openapi_client.types.runs_filter_data_source_type_enum', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/get_online_evaluator_spend_response.py': 'langsmith._openapi_client.types.get_online_evaluator_spend_response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/online_evaluator_bulk_delete_params.py': 'langsmith._openapi_client.types.online_evaluator_bulk_delete_params', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/types/__init__.py': 'langsmith._openapi_client.types', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_exceptions.py': 'langsmith._openapi_client._exceptions', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_qs.py': 'langsmith._openapi_client._qs', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_version.py': 'langsmith._openapi_client._version', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_streaming.py': 'langsmith._openapi_client._streaming', '/usr/local/lib/python3.10/site-packages/distro/distro.py': 'distro.distro', '/usr/local/lib/python3.10/site-packages/distro/__init__.py': 'distro', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_response.py': 'langsmith._openapi_client._response', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_json.py': 'langsmith._openapi_client._utils._json', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_base_client.py': 'langsmith._openapi_client._base_client', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_client.py': 'langsmith._openapi_client._client', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_logs.py': 'langsmith._openapi_client._utils._logs', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/_utils/_resources_proxy.py': 'langsmith._openapi_client._utils._resources_proxy', '/usr/local/lib/python3.10/site-packages/langsmith/_openapi_client/__init__.py': 'langsmith._openapi_client', '/usr/local/lib/python3.10/site-packages/langsmith/prompt_cache.py': 'langsmith.prompt_cache', '/usr/local/lib/python3.10/site-packages/langsmith/client.py': 'langsmith.client', '/usr/local/lib/python3.10/site-packages/langsmith/uuid.py': 'langsmith.uuid', '/usr/local/lib/python3.10/site-packages/langsmith/run_trees.py': 'langsmith.run_trees', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/schemas.py': 'langchain_core.tracers.schemas', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/core.py': 'langchain_core.tracers.core', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/base.py': 'langchain_core.tracers.base', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/memory_stream.py': 'langchain_core.tracers.memory_stream', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/log_stream.py': 'langchain_core.tracers.log_stream', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/aiter.py': 'langchain_core.utils.aiter', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/event_stream.py': 'langchain_core.tracers.event_stream', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/root_listeners.py': 'langchain_core.tracers.root_listeners', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/iter.py': 'langchain_core.utils.iter', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/base.py': 'langchain_core.runnables.base', '/usr/local/lib/python3.10/site-packages/transformers/dependency_versions_table.py': 'transformers.dependency_versions_table', '/usr/local/lib/python3.10/site-packages/huggingface_hub/__init__.py': 'huggingface_hub', '/usr/local/lib/python3.10/site-packages/tqdm/_monitor.py': 'tqdm._monitor', '/usr/local/lib/python3.10/site-packages/tqdm/_tqdm_pandas.py': 'tqdm._tqdm_pandas', '/usr/local/lib/python3.10/site-packages/tqdm/utils.py': 'tqdm.utils', '/usr/local/lib/python3.10/site-packages/tqdm/std.py': 'tqdm.std', '/usr/local/lib/python3.10/site-packages/tqdm/version.py': 'tqdm.version', '/usr/local/lib/python3.10/site-packages/tqdm/cli.py': 'tqdm.cli', '/usr/local/lib/python3.10/site-packages/tqdm/gui.py': 'tqdm.gui', '/usr/local/lib/python3.10/site-packages/tqdm/__init__.py': 'tqdm', '/usr/local/lib/python3.10/site-packages/tqdm/autonotebook.py': 'tqdm.autonotebook', '/usr/local/lib/python3.10/site-packages/tqdm/asyncio.py': 'tqdm.asyncio', '/usr/local/lib/python3.10/site-packages/tqdm/auto.py': 'tqdm.auto', '/usr/local/lib/python3.10/site-packages/tqdm/contrib/__init__.py': 'tqdm.contrib', '/usr/local/lib/python3.10/site-packages/tqdm/contrib/concurrent.py': 'tqdm.contrib.concurrent', '/usr/local/lib/python3.10/site-packages/huggingface_hub/constants.py': 'huggingface_hub.constants', '/usr/local/lib/python3.10/site-packages/huggingface_hub/errors.py': 'huggingface_hub.errors', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/tqdm.py': 'huggingface_hub.utils.tqdm', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_runtime.py': 'huggingface_hub.utils._runtime', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_auth.py': 'huggingface_hub.utils._auth', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_cache_assets.py': 'huggingface_hub.utils._cache_assets', '/usr/local/lib/python3.10/site-packages/huggingface_hub/commands/__init__.py': 'huggingface_hub.commands', '/usr/local/lib/python3.10/site-packages/huggingface_hub/commands/_cli_utils.py': 'huggingface_hub.commands._cli_utils', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/logging.py': 'huggingface_hub.utils.logging', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_cache_manager.py': 'huggingface_hub.utils._cache_manager', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_chunk_utils.py': 'huggingface_hub.utils._chunk_utils', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_datetime.py': 'huggingface_hub.utils._datetime', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_experimental.py': 'huggingface_hub.utils._experimental', '/usr/local/lib/python3.10/site-packages/filelock/_error.py': 'filelock._error', '/usr/local/lib/python3.10/site-packages/filelock/_util.py': 'filelock._util', '/usr/local/lib/python3.10/site-packages/filelock/_api.py': 'filelock._api', '/usr/local/lib/python3.10/site-packages/filelock/_unix.py': 'filelock._unix', '/usr/local/lib/python3.10/site-packages/filelock/_descriptor.py': 'filelock._descriptor', '/usr/local/lib/python3.10/encodings/ascii.py': 'encodings.ascii', '/usr/local/lib/python3.10/site-packages/filelock/_identity.py': 'filelock._identity', '/usr/local/lib/python3.10/site-packages/filelock/_soft_protocol.py': 'filelock._soft_protocol', '/usr/local/lib/python3.10/site-packages/filelock/_soft.py': 'filelock._soft', '/usr/local/lib/python3.10/site-packages/filelock/_marker.py': 'filelock._marker', '/usr/local/lib/python3.10/site-packages/filelock/_lease.py': 'filelock._lease', '/usr/local/lib/python3.10/lib-dynload/_sqlite3.cpython-310-x86_64-linux-gnu.so': '_sqlite3', '/usr/local/lib/python3.10/sqlite3/dbapi2.py': 'sqlite3.dbapi2', '/usr/local/lib/python3.10/sqlite3/__init__.py': 'sqlite3', '/usr/local/lib/python3.10/site-packages/filelock/_async.py': 'filelock._async', '/usr/local/lib/python3.10/site-packages/filelock/_read_write.py': 'filelock._read_write', '/usr/local/lib/python3.10/site-packages/filelock/_async_read_write.py': 'filelock._async_read_write', '/usr/local/lib/python3.10/site-packages/filelock/_soft_rw/_sync.py': 'filelock._soft_rw._sync', '/usr/local/lib/python3.10/site-packages/filelock/_soft_rw/_async.py': 'filelock._soft_rw._async', '/usr/local/lib/python3.10/site-packages/filelock/_soft_rw/__init__.py': 'filelock._soft_rw', '/usr/local/lib/python3.10/site-packages/filelock/_strict.py': 'filelock._strict', '/usr/local/lib/python3.10/site-packages/filelock/_windows.py': 'filelock._windows', '/usr/local/lib/python3.10/site-packages/filelock/asyncio.py': 'filelock.asyncio', '/usr/local/lib/python3.10/site-packages/filelock/version.py': 'filelock.version', '/usr/local/lib/python3.10/site-packages/filelock/__init__.py': 'filelock', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_fixes.py': 'huggingface_hub.utils._fixes', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_subprocess.py': 'huggingface_hub.utils._subprocess', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_git_credential.py': 'huggingface_hub.utils._git_credential', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_deprecation.py': 'huggingface_hub.utils._deprecation', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_typing.py': 'huggingface_hub.utils._typing', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_validators.py': 'huggingface_hub.utils._validators', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_headers.py': 'huggingface_hub.utils._headers', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_hf_folder.py': 'huggingface_hub.utils._hf_folder', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_lfs.py': 'huggingface_hub.utils._lfs', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_http.py': 'huggingface_hub.utils._http', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_pagination.py': 'huggingface_hub.utils._pagination', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_paths.py': 'huggingface_hub.utils._paths', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_safetensors.py': 'huggingface_hub.utils._safetensors', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_telemetry.py': 'huggingface_hub.utils._telemetry', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/_xet.py': 'huggingface_hub.utils._xet', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/__init__.py': 'huggingface_hub.utils', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_local_folder.py': 'huggingface_hub._local_folder', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/insecure_hashlib.py': 'huggingface_hub.utils.insecure_hashlib', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/sha.py': 'huggingface_hub.utils.sha', '/usr/local/lib/python3.10/site-packages/huggingface_hub/file_download.py': 'huggingface_hub.file_download', '/usr/local/lib/python3.10/site-packages/huggingface_hub/lfs.py': 'huggingface_hub.lfs', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_commit_api.py': 'huggingface_hub._commit_api', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_inference_endpoints.py': 'huggingface_hub._inference_endpoints', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_space_api.py': 'huggingface_hub._space_api', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_jobs_api.py': 'huggingface_hub._jobs_api', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_upload_large_folder.py': 'huggingface_hub._upload_large_folder', '/usr/local/lib/python3.10/site-packages/huggingface_hub/community.py': 'huggingface_hub.community', '/usr/local/lib/python3.10/site-packages/huggingface_hub/repocard_data.py': 'huggingface_hub.repocard_data', '/usr/local/lib/python3.10/site-packages/huggingface_hub/utils/endpoint_helpers.py': 'huggingface_hub.utils.endpoint_helpers', '/usr/local/lib/python3.10/site-packages/huggingface_hub/hf_api.py': 'huggingface_hub.hf_api', '/usr/local/lib/python3.10/site-packages/regex/_regex.cpython-310-x86_64-linux-gnu.so': 'regex._regex', '/usr/local/lib/python3.10/site-packages/regex/_regex_core.py': 'regex._regex_core', '/usr/local/lib/python3.10/site-packages/regex/_main.py': 'regex._main', '/usr/local/lib/python3.10/site-packages/regex/__init__.py': 'regex', '/usr/local/lib/python3.10/site-packages/transformers/utils/doc.py': 'transformers.utils.doc', '/usr/local/lib/python3.10/site-packages/numpy/_utils/_convertions.py': 'numpy._utils._convertions', '/usr/local/lib/python3.10/site-packages/numpy/_utils/__init__.py': 'numpy._utils', '/usr/local/lib/python3.10/site-packages/numpy/_globals.py': 'numpy._globals', '/usr/local/lib/python3.10/site-packages/numpy/_expired_attrs_2_0.py': 'numpy._expired_attrs_2_0', '/usr/local/lib/python3.10/site-packages/numpy/version.py': 'numpy.version', '/usr/local/lib/python3.10/site-packages/numpy/_distributor_init.py': 'numpy._distributor_init', '/usr/local/lib/python3.10/site-packages/numpy/_utils/_inspect.py': 'numpy._utils._inspect', '/usr/local/lib/python3.10/site-packages/numpy/exceptions.py': 'numpy.exceptions', '/usr/local/lib/python3.10/site-packages/numpy/_core/_exceptions.py': 'numpy._core._exceptions', '/usr/local/lib/python3.10/site-packages/numpy/_core/printoptions.py': 'numpy._core.printoptions', '/usr/local/lib/python3.10/site-packages/numpy/dtypes.py': 'numpy.dtypes', '/usr/local/lib/python3.10/site-packages/numpy/_core/_multiarray_umath.cpython-310-x86_64-linux-gnu.so': 'numpy._core._multiarray_umath', '/usr/local/lib/python3.10/site-packages/numpy/_core/overrides.py': 'numpy._core.overrides', '/usr/local/lib/python3.10/site-packages/numpy/_core/multiarray.py': 'numpy._core.multiarray', '/usr/local/lib/python3.10/site-packages/numpy/_core/umath.py': 'numpy._core.umath', '/usr/local/lib/python3.10/site-packages/numpy/_core/_string_helpers.py': 'numpy._core._string_helpers', '/usr/local/lib/python3.10/site-packages/numpy/_core/_type_aliases.py': 'numpy._core._type_aliases', '/usr/local/lib/python3.10/site-packages/numpy/_core/_dtype.py': 'numpy._core._dtype', '/usr/local/lib/python3.10/site-packages/numpy/_core/numerictypes.py': 'numpy._core.numerictypes', '/usr/local/lib/python3.10/site-packages/numpy/_core/_methods.py': 'numpy._core._methods', '/usr/local/lib/python3.10/site-packages/numpy/_core/fromnumeric.py': 'numpy._core.fromnumeric', '/usr/local/lib/python3.10/site-packages/numpy/_core/shape_base.py': 'numpy._core.shape_base', '/usr/local/lib/python3.10/site-packages/numpy/_core/_ufunc_config.py': 'numpy._core._ufunc_config', '/usr/local/lib/python3.10/site-packages/numpy/_core/arrayprint.py': 'numpy._core.arrayprint', '/usr/local/lib/python3.10/site-packages/numpy/_core/_asarray.py': 'numpy._core._asarray', '/usr/local/lib/python3.10/site-packages/numpy/_core/numeric.py': 'numpy._core.numeric', '/usr/local/lib/python3.10/site-packages/numpy/_core/records.py': 'numpy._core.records', '/usr/local/lib/python3.10/site-packages/numpy/_core/memmap.py': 'numpy._core.memmap', '/usr/local/lib/python3.10/site-packages/numpy/_core/function_base.py': 'numpy._core.function_base', '/usr/local/lib/python3.10/site-packages/numpy/_core/_machar.py': 'numpy._core._machar', '/usr/local/lib/python3.10/site-packages/numpy/_core/getlimits.py': 'numpy._core.getlimits', '/usr/local/lib/python3.10/site-packages/numpy/_core/einsumfunc.py': 'numpy._core.einsumfunc', '/usr/local/lib/python3.10/site-packages/numpy/_core/_add_newdocs.py': 'numpy._core._add_newdocs', '/usr/local/lib/python3.10/site-packages/numpy/_core/_add_newdocs_scalars.py': 'numpy._core._add_newdocs_scalars', '/usr/local/lib/python3.10/site-packages/numpy/_core/_dtype_ctypes.py': 'numpy._core._dtype_ctypes', '/usr/local/lib/python3.10/site-packages/numpy/_core/_internal.py': 'numpy._core._internal', '/usr/local/lib/python3.10/site-packages/numpy/_pytesttester.py': 'numpy._pytesttester', '/usr/local/lib/python3.10/site-packages/numpy/_core/__init__.py': 'numpy._core', '/usr/local/lib/python3.10/site-packages/numpy/__config__.py': 'numpy.__config__', '/usr/local/lib/python3.10/site-packages/numpy/lib/_array_utils_impl.py': 'numpy.lib._array_utils_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/array_utils.py': 'numpy.lib.array_utils', '/usr/local/lib/python3.10/site-packages/numpy/lib/introspect.py': 'numpy.lib.introspect', '/usr/local/lib/python3.10/site-packages/numpy/lib/mixins.py': 'numpy.lib.mixins', '/usr/local/lib/python3.10/site-packages/numpy/lib/_utils_impl.py': 'numpy.lib._utils_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/format.py': 'numpy.lib.format', '/usr/local/lib/python3.10/site-packages/numpy/lib/_datasource.py': 'numpy.lib._datasource', '/usr/local/lib/python3.10/site-packages/numpy/lib/_iotools.py': 'numpy.lib._iotools', '/usr/local/lib/python3.10/site-packages/numpy/lib/_npyio_impl.py': 'numpy.lib._npyio_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/npyio.py': 'numpy.lib.npyio', '/usr/local/lib/python3.10/site-packages/numpy/lib/_ufunclike_impl.py': 'numpy.lib._ufunclike_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_type_check_impl.py': 'numpy.lib._type_check_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_scimath_impl.py': 'numpy.lib._scimath_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/scimath.py': 'numpy.lib.scimath', '/usr/local/lib/python3.10/site-packages/numpy/lib/_stride_tricks_impl.py': 'numpy.lib._stride_tricks_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/stride_tricks.py': 'numpy.lib.stride_tricks', '/usr/local/lib/python3.10/site-packages/numpy/linalg/linalg.py': 'numpy.linalg.linalg', '/usr/local/lib/python3.10/site-packages/numpy/lib/_twodim_base_impl.py': 'numpy.lib._twodim_base_impl', '/usr/local/lib/python3.10/site-packages/numpy/linalg/_umath_linalg.cpython-310-x86_64-linux-gnu.so': 'numpy.linalg._umath_linalg', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_nested_sequence.py': 'numpy._typing._nested_sequence', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_nbit_base.py': 'numpy._typing._nbit_base', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_nbit.py': 'numpy._typing._nbit', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_char_codes.py': 'numpy._typing._char_codes', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_scalars.py': 'numpy._typing._scalars', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_shape.py': 'numpy._typing._shape', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_dtype_like.py': 'numpy._typing._dtype_like', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_array_like.py': 'numpy._typing._array_like', '/usr/local/lib/python3.10/site-packages/numpy/_typing/_ufunc.py': 'numpy._typing._ufunc', '/usr/local/lib/python3.10/site-packages/numpy/_typing/__init__.py': 'numpy._typing', '/usr/local/lib/python3.10/site-packages/numpy/linalg/_linalg.py': 'numpy.linalg._linalg', '/usr/local/lib/python3.10/site-packages/numpy/linalg/__init__.py': 'numpy.linalg', '/usr/local/lib/python3.10/site-packages/numpy/matrixlib/defmatrix.py': 'numpy.matrixlib.defmatrix', '/usr/local/lib/python3.10/site-packages/numpy/matrixlib/__init__.py': 'numpy.matrixlib', '/usr/local/lib/python3.10/site-packages/numpy/lib/_histograms_impl.py': 'numpy.lib._histograms_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_function_base_impl.py': 'numpy.lib._function_base_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_index_tricks_impl.py': 'numpy.lib._index_tricks_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_nanfunctions_impl.py': 'numpy.lib._nanfunctions_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_shape_base_impl.py': 'numpy.lib._shape_base_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_arraysetops_impl.py': 'numpy.lib._arraysetops_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_polynomial_impl.py': 'numpy.lib._polynomial_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_arrayterator_impl.py': 'numpy.lib._arrayterator_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_arraypad_impl.py': 'numpy.lib._arraypad_impl', '/usr/local/lib/python3.10/site-packages/numpy/lib/_version.py': 'numpy.lib._version', '/usr/local/lib/python3.10/site-packages/numpy/lib/__init__.py': 'numpy.lib', '/usr/local/lib/python3.10/site-packages/numpy/_array_api_info.py': 'numpy._array_api_info', '/usr/local/lib/python3.10/site-packages/numpy/__init__.py': 'numpy', '/usr/local/lib/python3.10/site-packages/transformers/utils/logging.py': 'transformers.utils.logging', '/usr/local/lib/python3.10/site-packages/transformers/utils/import_utils.py': 'transformers.utils.import_utils', '/usr/local/lib/python3.10/site-packages/transformers/utils/generic.py': 'transformers.utils.generic', '/usr/local/lib/python3.10/site-packages/transformers/utils/auto_docstring.py': 'transformers.utils.auto_docstring', '/usr/local/lib/python3.10/site-packages/transformers/utils/backbone_utils.py': 'transformers.utils.backbone_utils', '/usr/local/lib/python3.10/site-packages/jinja2/bccache.py': 'jinja2.bccache', '/usr/local/lib/python3.10/site-packages/markupsafe/_speedups.cpython-310-x86_64-linux-gnu.so': 'markupsafe._speedups', '/usr/local/lib/python3.10/site-packages/markupsafe/__init__.py': 'markupsafe', '/usr/local/lib/python3.10/site-packages/jinja2/utils.py': 'jinja2.utils', '/usr/local/lib/python3.10/site-packages/jinja2/nodes.py': 'jinja2.nodes', '/usr/local/lib/python3.10/site-packages/jinja2/exceptions.py': 'jinja2.exceptions', '/usr/local/lib/python3.10/site-packages/jinja2/visitor.py': 'jinja2.visitor', '/usr/local/lib/python3.10/site-packages/jinja2/idtracking.py': 'jinja2.idtracking', '/usr/local/lib/python3.10/site-packages/jinja2/optimizer.py': 'jinja2.optimizer', '/usr/local/lib/python3.10/site-packages/jinja2/compiler.py': 'jinja2.compiler', '/usr/local/lib/python3.10/site-packages/jinja2/async_utils.py': 'jinja2.async_utils', '/usr/local/lib/python3.10/site-packages/jinja2/runtime.py': 'jinja2.runtime', '/usr/local/lib/python3.10/site-packages/jinja2/filters.py': 'jinja2.filters', '/usr/local/lib/python3.10/site-packages/jinja2/tests.py': 'jinja2.tests', '/usr/local/lib/python3.10/site-packages/jinja2/defaults.py': 'jinja2.defaults', '/usr/local/lib/python3.10/site-packages/jinja2/_identifier.py': 'jinja2._identifier', '/usr/local/lib/python3.10/site-packages/jinja2/lexer.py': 'jinja2.lexer', '/usr/local/lib/python3.10/site-packages/jinja2/parser.py': 'jinja2.parser', '/usr/local/lib/python3.10/site-packages/jinja2/environment.py': 'jinja2.environment', '/usr/local/lib/python3.10/site-packages/jinja2/loaders.py': 'jinja2.loaders', '/usr/local/lib/python3.10/site-packages/jinja2/__init__.py': 'jinja2', '/usr/local/lib/python3.10/site-packages/jinja2/ext.py': 'jinja2.ext', '/usr/local/lib/python3.10/site-packages/jinja2/sandbox.py': 'jinja2.sandbox', '/usr/local/lib/python3.10/site-packages/PIL/_version.py': 'PIL._version', '/usr/local/lib/python3.10/site-packages/PIL/__init__.py': 'PIL', '/usr/local/lib/python3.10/site-packages/PIL/ExifTags.py': 'PIL.ExifTags', '/usr/local/lib/python3.10/site-packages/PIL/ImageMode.py': 'PIL.ImageMode', '/usr/local/lib/python3.10/site-packages/PIL/TiffTags.py': 'PIL.TiffTags', '/usr/local/lib/python3.10/site-packages/PIL/_binary.py': 'PIL._binary', '/usr/local/lib/python3.10/site-packages/PIL/_deprecate.py': 'PIL._deprecate', '/usr/local/lib/python3.10/site-packages/PIL/_util.py': 'PIL._util', '/usr/local/lib/python3.10/xml/parsers/__init__.py': 'xml.parsers', '/usr/local/lib/python3.10/xml/parsers/expat.py': 'xml.parsers.expat', '/usr/local/lib/python3.10/site-packages/defusedxml/common.py': 'defusedxml.common', '/usr/local/lib/python3.10/site-packages/defusedxml/__init__.py': 'defusedxml', '/usr/local/lib/python3.10/lib-dynload/_elementtree.cpython-310-x86_64-linux-gnu.so': '_elementtree', '/usr/local/lib/python3.10/xml/etree/ElementTree.py': 'xml.etree.ElementTree', '/usr/local/lib/python3.10/site-packages/defusedxml/ElementTree.py': 'defusedxml.ElementTree', '/usr/local/lib/python3.10/site-packages/PIL/_imaging.cpython-310-x86_64-linux-gnu.so': 'PIL._imaging', '/usr/local/lib/python3.10/site-packages/PIL/Image.py': 'PIL.Image', '/usr/local/lib/python3.10/site-packages/transformers/utils/chat_template_utils.py': 'transformers.utils.chat_template_utils', '/usr/local/lib/python3.10/site-packages/transformers/utils/constants.py': 'transformers.utils.constants', '/usr/local/lib/python3.10/site-packages/huggingface_hub/repocard.py': 'huggingface_hub.repocard', '/usr/local/lib/python3.10/site-packages/huggingface_hub/_snapshot_download.py': 'huggingface_hub._snapshot_download', '/usr/local/lib/python3.10/site-packages/transformers/utils/hub.py': 'transformers.utils.hub', '/usr/local/lib/python3.10/site-packages/transformers/utils/peft_utils.py': 'transformers.utils.peft_utils', '/usr/local/lib/python3.10/site-packages/transformers/utils/__init__.py': 'transformers.utils', '/usr/local/lib/python3.10/site-packages/transformers/utils/versions.py': 'transformers.utils.versions', '/usr/local/lib/python3.10/site-packages/transformers/dependency_versions_check.py': 'transformers.dependency_versions_check', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_sentencepiece_and_tokenizers_objects.py': 'transformers.utils.dummy_sentencepiece_and_tokenizers_objects', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_mistral_common_objects.py': 'transformers.utils.dummy_mistral_common_objects', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_torchvision_objects.py': 'transformers.utils.dummy_torchvision_objects', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_pt_objects.py': 'transformers.utils.dummy_pt_objects', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_tf_objects.py': 'transformers.utils.dummy_tf_objects', '/usr/local/lib/python3.10/site-packages/transformers/utils/dummy_flax_objects.py': 'transformers.utils.dummy_flax_objects', '/usr/local/lib/python3.10/site-packages/transformers/__init__.py': 'transformers', '/usr/local/lib/python3.10/site-packages/transformers/models/__init__.py': 'transformers.models', '/usr/local/lib/python3.10/site-packages/transformers/models/gpt2/__init__.py': 'transformers.models.gpt2', '/usr/local/lib/python3.10/filecmp.py': 'filecmp', '/usr/local/lib/python3.10/site-packages/transformers/dynamic_module_utils.py': 'transformers.dynamic_module_utils', '/usr/local/lib/python3.10/site-packages/tokenizers/tokenizers.abi3.so': 'tokenizers.tokenizers', '/usr/local/lib/python3.10/site-packages/tokenizers/decoders/__init__.py': 'tokenizers.decoders', '/usr/local/lib/python3.10/site-packages/tokenizers/models/__init__.py': 'tokenizers.models', '/usr/local/lib/python3.10/site-packages/tokenizers/normalizers/__init__.py': 'tokenizers.normalizers', '/usr/local/lib/python3.10/site-packages/tokenizers/pre_tokenizers/__init__.py': 'tokenizers.pre_tokenizers', '/usr/local/lib/python3.10/site-packages/tokenizers/processors/__init__.py': 'tokenizers.processors', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/base_tokenizer.py': 'tokenizers.implementations.base_tokenizer', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/bert_wordpiece.py': 'tokenizers.implementations.bert_wordpiece', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/byte_level_bpe.py': 'tokenizers.implementations.byte_level_bpe', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/char_level_bpe.py': 'tokenizers.implementations.char_level_bpe', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/sentencepiece_bpe.py': 'tokenizers.implementations.sentencepiece_bpe', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/sentencepiece_unigram.py': 'tokenizers.implementations.sentencepiece_unigram', '/usr/local/lib/python3.10/site-packages/tokenizers/implementations/__init__.py': 'tokenizers.implementations', '/usr/local/lib/python3.10/site-packages/tokenizers/__init__.py': 'tokenizers', '/usr/local/lib/python3.10/site-packages/transformers/tokenization_utils_base.py': 'transformers.tokenization_utils_base', '/usr/local/lib/python3.10/site-packages/tokenizers/trainers/__init__.py': 'tokenizers.trainers', '/usr/local/lib/python3.10/site-packages/transformers/convert_slow_tokenizer.py': 'transformers.convert_slow_tokenizer', '/usr/local/lib/python3.10/site-packages/transformers/integrations/__init__.py': 'transformers.integrations', '/usr/local/lib/python3.10/site-packages/transformers/integrations/ggml.py': 'transformers.integrations.ggml', '/usr/local/lib/python3.10/site-packages/transformers/modeling_gguf_pytorch_utils.py': 'transformers.modeling_gguf_pytorch_utils', '/usr/local/lib/python3.10/site-packages/transformers/tokenization_utils.py': 'transformers.tokenization_utils', '/usr/local/lib/python3.10/site-packages/transformers/tokenization_utils_fast.py': 'transformers.tokenization_utils_fast', '/usr/local/lib/python3.10/site-packages/transformers/models/gpt2/tokenization_gpt2.py': 'transformers.models.gpt2.tokenization_gpt2', '/usr/local/lib/python3.10/site-packages/transformers/models/gpt2/tokenization_gpt2_fast.py': 'transformers.models.gpt2.tokenization_gpt2_fast', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/base.py': 'langchain_core.language_models.base', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/base.py': 'langchain_core.output_parsers.base', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/base.py': 'langchain_core.prompts.base', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/mustache.py': 'langchain_core.utils.mustache', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/formatting.py': 'langchain_core.utils.formatting', '/usr/local/lib/python3.10/site-packages/jinja2/meta.py': 'jinja2.meta', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/string.py': 'langchain_core.prompts.string', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/dict.py': 'langchain_core.prompts.dict', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/image.py': 'langchain_core.prompts.image', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/message.py': 'langchain_core.prompts.message', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/prompt.py': 'langchain_core.prompts.prompt', '/usr/local/lib/python3.10/site-packages/langchain_core/prompts/chat.py': 'langchain_core.prompts.chat', '/usr/local/lib/python3.10/site-packages/antlr4/Token.py': 'antlr4.Token', '/usr/local/lib/python3.10/site-packages/antlr4/InputStream.py': 'antlr4.InputStream', '/usr/local/lib/python3.10/site-packages/antlr4/FileStream.py': 'antlr4.FileStream', '/usr/local/lib/python3.10/site-packages/antlr4/StdinStream.py': 'antlr4.StdinStream', '/usr/local/lib/python3.10/site-packages/antlr4/error/__init__.py': 'antlr4.error', '/usr/local/lib/python3.10/site-packages/antlr4/tree/__init__.py': 'antlr4.tree', '/usr/local/lib/python3.10/site-packages/antlr4/tree/Tree.py': 'antlr4.tree.Tree', '/usr/local/lib/python3.10/site-packages/antlr4/Utils.py': 'antlr4.Utils', '/usr/local/lib/python3.10/site-packages/antlr4/tree/Trees.py': 'antlr4.tree.Trees', '/usr/local/lib/python3.10/site-packages/antlr4/RuleContext.py': 'antlr4.RuleContext', '/usr/local/lib/python3.10/site-packages/antlr4/ParserRuleContext.py': 'antlr4.ParserRuleContext', '/usr/local/lib/python3.10/site-packages/antlr4/error/ErrorListener.py': 'antlr4.error.ErrorListener', '/usr/local/lib/python3.10/site-packages/antlr4/Recognizer.py': 'antlr4.Recognizer', '/usr/local/lib/python3.10/site-packages/antlr4/error/Errors.py': 'antlr4.error.Errors', '/usr/local/lib/python3.10/site-packages/antlr4/BufferedTokenStream.py': 'antlr4.BufferedTokenStream', '/usr/local/lib/python3.10/site-packages/antlr4/CommonTokenFactory.py': 'antlr4.CommonTokenFactory', '/usr/local/lib/python3.10/site-packages/antlr4/atn/__init__.py': 'antlr4.atn', '/usr/local/lib/python3.10/site-packages/antlr4/IntervalSet.py': 'antlr4.IntervalSet', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNType.py': 'antlr4.atn.ATNType', '/usr/local/lib/python3.10/site-packages/antlr4/atn/SemanticContext.py': 'antlr4.atn.SemanticContext', '/usr/local/lib/python3.10/site-packages/antlr4/atn/Transition.py': 'antlr4.atn.Transition', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNState.py': 'antlr4.atn.ATNState', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATN.py': 'antlr4.atn.ATN', '/usr/local/lib/python3.10/site-packages/antlr4/PredictionContext.py': 'antlr4.PredictionContext', '/usr/local/lib/python3.10/site-packages/antlr4/atn/LexerAction.py': 'antlr4.atn.LexerAction', '/usr/local/lib/python3.10/site-packages/antlr4/atn/LexerActionExecutor.py': 'antlr4.atn.LexerActionExecutor', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNConfig.py': 'antlr4.atn.ATNConfig', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNConfigSet.py': 'antlr4.atn.ATNConfigSet', '/usr/local/lib/python3.10/site-packages/antlr4/dfa/__init__.py': 'antlr4.dfa', '/usr/local/lib/python3.10/site-packages/antlr4/dfa/DFAState.py': 'antlr4.dfa.DFAState', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNSimulator.py': 'antlr4.atn.ATNSimulator', '/usr/local/lib/python3.10/site-packages/antlr4/atn/LexerATNSimulator.py': 'antlr4.atn.LexerATNSimulator', '/usr/local/lib/python3.10/site-packages/antlr4/Lexer.py': 'antlr4.Lexer', '/usr/local/lib/python3.10/site-packages/antlr4/CommonTokenStream.py': 'antlr4.CommonTokenStream', '/usr/local/lib/python3.10/site-packages/antlr4/error/ErrorStrategy.py': 'antlr4.error.ErrorStrategy', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNDeserializationOptions.py': 'antlr4.atn.ATNDeserializationOptions', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ATNDeserializer.py': 'antlr4.atn.ATNDeserializer', '/usr/local/lib/python3.10/site-packages/antlr4/ListTokenSource.py': 'antlr4.ListTokenSource', '/usr/local/lib/python3.10/site-packages/antlr4/tree/Chunk.py': 'antlr4.tree.Chunk', '/usr/local/lib/python3.10/site-packages/antlr4/tree/RuleTagToken.py': 'antlr4.tree.RuleTagToken', '/usr/local/lib/python3.10/site-packages/antlr4/tree/TokenTagToken.py': 'antlr4.tree.TokenTagToken', '/usr/local/lib/python3.10/site-packages/antlr4/tree/ParseTreePatternMatcher.py': 'antlr4.tree.ParseTreePatternMatcher', '/usr/local/lib/python3.10/site-packages/antlr4/Parser.py': 'antlr4.Parser', '/usr/local/lib/python3.10/site-packages/antlr4/dfa/DFA.py': 'antlr4.dfa.DFA', '/usr/local/lib/python3.10/site-packages/antlr4/atn/PredictionMode.py': 'antlr4.atn.PredictionMode', '/usr/local/lib/python3.10/site-packages/antlr4/atn/ParserATNSimulator.py': 'antlr4.atn.ParserATNSimulator', '/usr/local/lib/python3.10/site-packages/antlr4/error/DiagnosticErrorListener.py': 'antlr4.error.DiagnosticErrorListener', '/usr/local/lib/python3.10/site-packages/antlr4/__init__.py': 'antlr4', '/usr/local/lib/python3.10/site-packages/omegaconf/errors.py': 'omegaconf.errors', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar/__init__.py': 'omegaconf.grammar', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar/gen/__init__.py': 'omegaconf.grammar.gen', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar/gen/OmegaConfGrammarLexer.py': 'omegaconf.grammar.gen.OmegaConfGrammarLexer', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar/gen/OmegaConfGrammarParser.py': 'omegaconf.grammar.gen.OmegaConfGrammarParser', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar/gen/OmegaConfGrammarParserVisitor.py': 'omegaconf.grammar.gen.OmegaConfGrammarParserVisitor', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar_visitor.py': 'omegaconf.grammar_visitor', '/usr/local/lib/python3.10/site-packages/omegaconf/grammar_parser.py': 'omegaconf.grammar_parser', '/usr/local/lib/python3.10/site-packages/attr/_compat.py': 'attr._compat', '/usr/local/lib/python3.10/site-packages/attr/_config.py': 'attr._config', '/usr/local/lib/python3.10/site-packages/attr/exceptions.py': 'attr.exceptions', '/usr/local/lib/python3.10/site-packages/attr/setters.py': 'attr.setters', '/usr/local/lib/python3.10/site-packages/attr/_make.py': 'attr._make', '/usr/local/lib/python3.10/site-packages/attr/converters.py': 'attr.converters', '/usr/local/lib/python3.10/site-packages/attr/filters.py': 'attr.filters', '/usr/local/lib/python3.10/site-packages/attr/validators.py': 'attr.validators', '/usr/local/lib/python3.10/site-packages/attr/_cmp.py': 'attr._cmp', '/usr/local/lib/python3.10/site-packages/attr/_funcs.py': 'attr._funcs', '/usr/local/lib/python3.10/site-packages/attr/_next_gen.py': 'attr._next_gen', '/usr/local/lib/python3.10/site-packages/attr/_version_info.py': 'attr._version_info', '/usr/local/lib/python3.10/site-packages/attr/__init__.py': 'attr', '/usr/local/lib/python3.10/site-packages/omegaconf/_utils.py': 'omegaconf._utils', '/usr/local/lib/python3.10/site-packages/omegaconf/base.py': 'omegaconf.base', '/usr/local/lib/python3.10/site-packages/omegaconf/basecontainer.py': 'omegaconf.basecontainer', '/usr/local/lib/python3.10/site-packages/omegaconf/nodes.py': 'omegaconf.nodes', '/usr/local/lib/python3.10/site-packages/omegaconf/dictconfig.py': 'omegaconf.dictconfig', '/usr/local/lib/python3.10/site-packages/omegaconf/listconfig.py': 'omegaconf.listconfig', '/usr/local/lib/python3.10/site-packages/omegaconf/resolvers/oc/dict.py': 'omegaconf.resolvers.oc.dict', '/usr/local/lib/python3.10/site-packages/omegaconf/resolvers/oc/__init__.py': 'omegaconf.resolvers.oc', '/usr/local/lib/python3.10/site-packages/omegaconf/resolvers/__init__.py': 'omegaconf.resolvers', '/usr/local/lib/python3.10/site-packages/omegaconf/omegaconf.py': 'omegaconf.omegaconf', '/usr/local/lib/python3.10/site-packages/omegaconf/version.py': 'omegaconf.version', '/usr/local/lib/python3.10/site-packages/omegaconf/__init__.py': 'omegaconf', '/usr/local/lib/python3.10/site-packages/transformers/models/auto/__init__.py': 'transformers.models.auto', '/usr/local/lib/python3.10/site-packages/transformers/configuration_utils.py': 'transformers.configuration_utils', '/usr/local/lib/python3.10/site-packages/transformers/models/encoder_decoder/__init__.py': 'transformers.models.encoder_decoder', '/usr/local/lib/python3.10/site-packages/transformers/models/auto/configuration_auto.py': 'transformers.models.auto.configuration_auto', '/usr/local/lib/python3.10/site-packages/transformers/models/encoder_decoder/configuration_encoder_decoder.py': 'transformers.models.encoder_decoder.configuration_encoder_decoder', '/usr/local/lib/python3.10/site-packages/transformers/models/auto/auto_factory.py': 'transformers.models.auto.auto_factory', '/usr/local/lib/python3.10/site-packages/transformers/models/auto/tokenization_auto.py': 'transformers.models.auto.tokenization_auto', '/usr/local/lib/python3.10/site-packages/transformers/models/llama/__init__.py': 'transformers.models.llama', '/usr/local/lib/python3.10/site-packages/transformers/models/llama/tokenization_llama_fast.py': 'transformers.models.llama.tokenization_llama_fast', '/usr/local/lib/python3.10/site-packages/transformers/models/arcee/__init__.py': 'transformers.models.arcee', '/usr/local/lib/python3.10/site-packages/transformers/models/aria/__init__.py': 'transformers.models.aria', '/usr/local/lib/python3.10/site-packages/transformers/models/chameleon/__init__.py': 'transformers.models.chameleon', '/usr/local/lib/python3.10/site-packages/transformers/models/colpali/__init__.py': 'transformers.models.colpali', '/usr/local/lib/python3.10/site-packages/transformers/models/deepseek_v2/__init__.py': 'transformers.models.deepseek_v2', '/usr/local/lib/python3.10/site-packages/transformers/models/deepseek_v3/__init__.py': 'transformers.models.deepseek_v3', '/usr/local/lib/python3.10/site-packages/transformers/models/deepseek_vl/__init__.py': 'transformers.models.deepseek_vl', '/usr/local/lib/python3.10/site-packages/transformers/models/deepseek_vl_hybrid/__init__.py': 'transformers.models.deepseek_vl_hybrid', '/usr/local/lib/python3.10/site-packages/transformers/models/diffllama/__init__.py': 'transformers.models.diffllama', '/usr/local/lib/python3.10/site-packages/transformers/models/ernie4_5/__init__.py': 'transformers.models.ernie4_5', '/usr/local/lib/python3.10/site-packages/transformers/models/ernie4_5_moe/__init__.py': 'transformers.models.ernie4_5_moe', '/usr/local/lib/python3.10/site-packages/transformers/models/idefics/__init__.py': 'transformers.models.idefics', '/usr/local/lib/python3.10/site-packages/transformers/models/idefics2/__init__.py': 'transformers.models.idefics2', '/usr/local/lib/python3.10/site-packages/transformers/models/idefics3/__init__.py': 'transformers.models.idefics3', '/usr/local/lib/python3.10/site-packages/transformers/models/jamba/__init__.py': 'transformers.models.jamba', '/usr/local/lib/python3.10/site-packages/transformers/models/janus/__init__.py': 'transformers.models.janus', '/usr/local/lib/python3.10/site-packages/transformers/models/jetmoe/__init__.py': 'transformers.models.jetmoe', '/workspace/pynguin/utils/deepseek/deepseek_tokenizer.py': 'pynguin.utils.deepseek.deepseek_tokenizer', '/workspace/pynguin/utils/deepseek/__init__.py': 'pynguin.utils.deepseek', '/workspace/pynguin/llm/abstractmodel.py': 'pynguin.llm.abstractmodel', '/workspace/pynguin/llm/codamosa/outputfixers.py': 'pynguin.llm.codamosa.outputfixers', '/workspace/pynguin/llm/codamosa/model.py': 'pynguin.llm.codamosa.model', '/usr/local/lib/python3.10/site-packages/pygments/formatters/html.py': 'pygments.formatters.html', '/workspace/pynguin/utils/report.py': 'pynguin.utils.report', '/workspace/pynguin/llm/codamosa/llmseeding.py': 'pynguin.llm.codamosa.llmseeding', '/workspace/pynguin/ga/algorithms/codamosaalgorithm.py': 'pynguin.ga.algorithms.codamosaalgorithm', '/workspace/pynguin/utils/controlflowdistance.py': 'pynguin.utils.controlflowdistance', '/workspace/pynguin/ga/coveragegoals.py': 'pynguin.ga.coveragegoals', '/workspace/pynguin/llm/deepmosa/stmtdeserializer_v2.py': 'pynguin.llm.deepmosa.stmtdeserializer_v2', '/workspace/pynguin/llm/deepmosa/outputfixers.py': 'pynguin.llm.deepmosa.outputfixers', '/usr/local/lib/python3.10/site-packages/antlr4/LL1Analyzer.py': 'antlr4.LL1Analyzer', '/usr/local/lib/python3.10/site-packages/langchain_protocol/protocol.py': 'langchain_protocol.protocol', '/usr/local/lib/python3.10/site-packages/langchain_protocol/__init__.py': 'langchain_protocol', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/_compat_bridge.py': 'langchain_core.language_models._compat_bridge', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/chat_model_stream.py': 'langchain_core.language_models.chat_model_stream', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/model_profile.py': 'langchain_core.language_models.model_profile', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/transform.py': 'langchain_core.output_parsers.transform', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/openai_tools.py': 'langchain_core.output_parsers.openai_tools', '/usr/local/lib/python3.10/site-packages/langchain_core/outputs/chat_result.py': 'langchain_core.outputs.chat_result', '/usr/local/lib/python3.10/site-packages/langchain_core/rate_limiters.py': 'langchain_core.rate_limiters', '/usr/local/lib/python3.10/site-packages/langchain_core/runnables/passthrough.py': 'langchain_core.runnables.passthrough', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/json_schema.py': 'langchain_core.utils.json_schema', '/usr/local/lib/python3.10/site-packages/langchain_core/utils/function_calling.py': 'langchain_core.utils.function_calling', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/chat_models.py': 'langchain_core.language_models.chat_models', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/format_instructions.py': 'langchain_core.output_parsers.format_instructions', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/json.py': 'langchain_core.output_parsers.json', '/usr/local/lib/python3.10/site-packages/langchain_core/output_parsers/pydantic.py': 'langchain_core.output_parsers.pydantic', '/usr/local/lib/python3.10/site-packages/langchain_core/tools/__init__.py': 'langchain_core.tools', '/usr/local/lib/python3.10/site-packages/pydantic/alias_generators.py': 'pydantic.alias_generators', '/usr/local/lib/python3.10/site-packages/pydantic/deprecated/decorator.py': 'pydantic.deprecated.decorator', '/usr/local/lib/python3.10/site-packages/langchain_core/tools/base.py': 'langchain_core.tools.base', '/usr/local/lib/python3.10/site-packages/pydantic/functional_serializers.py': 'pydantic.functional_serializers', '/usr/local/lib/python3.10/site-packages/ollama/_types.py': 'ollama._types', '/usr/local/lib/python3.10/site-packages/ollama/_utils.py': 'ollama._utils', '/usr/local/lib/python3.10/site-packages/ollama/_client.py': 'ollama._client', '/usr/local/lib/python3.10/site-packages/httpcore/_models.py': 'httpcore._models', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/__init__.py': 'httpcore._backends', '/usr/local/lib/python3.10/site-packages/httpcore/_exceptions.py': 'httpcore._exceptions', '/usr/local/lib/python3.10/site-packages/httpcore/_utils.py': 'httpcore._utils', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/base.py': 'httpcore._backends.base', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/sync.py': 'httpcore._backends.sync', '/usr/local/lib/python3.10/site-packages/httpcore/_ssl.py': 'httpcore._ssl', '/usr/local/lib/python3.10/site-packages/httpcore/_synchronization.py': 'httpcore._synchronization', '/usr/local/lib/python3.10/site-packages/httpcore/_trace.py': 'httpcore._trace', '/usr/local/lib/python3.10/site-packages/h11/_abnf.py': 'h11._abnf', '/usr/local/lib/python3.10/site-packages/h11/_util.py': 'h11._util', '/usr/local/lib/python3.10/site-packages/h11/_headers.py': 'h11._headers', '/usr/local/lib/python3.10/site-packages/h11/_events.py': 'h11._events', '/usr/local/lib/python3.10/site-packages/h11/_receivebuffer.py': 'h11._receivebuffer', '/usr/local/lib/python3.10/site-packages/h11/_state.py': 'h11._state', '/usr/local/lib/python3.10/site-packages/h11/_readers.py': 'h11._readers', '/usr/local/lib/python3.10/site-packages/h11/_writers.py': 'h11._writers', '/usr/local/lib/python3.10/site-packages/h11/_connection.py': 'h11._connection', '/usr/local/lib/python3.10/site-packages/h11/_version.py': 'h11._version', '/usr/local/lib/python3.10/site-packages/h11/__init__.py': 'h11', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/interfaces.py': 'httpcore._sync.interfaces', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/http11.py': 'httpcore._sync.http11', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/connection.py': 'httpcore._sync.connection', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/connection_pool.py': 'httpcore._sync.connection_pool', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/http_proxy.py': 'httpcore._sync.http_proxy', '/usr/local/lib/python3.10/site-packages/httpcore/_sync/__init__.py': 'httpcore._sync', '/usr/local/lib/python3.10/site-packages/httpcore/_api.py': 'httpcore._api', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/auto.py': 'httpcore._backends.auto', '/usr/local/lib/python3.10/site-packages/httpcore/_async/interfaces.py': 'httpcore._async.interfaces', '/usr/local/lib/python3.10/site-packages/httpcore/_async/http11.py': 'httpcore._async.http11', '/usr/local/lib/python3.10/site-packages/httpcore/_async/connection.py': 'httpcore._async.connection', '/usr/local/lib/python3.10/site-packages/httpcore/_async/connection_pool.py': 'httpcore._async.connection_pool', '/usr/local/lib/python3.10/site-packages/httpcore/_async/http_proxy.py': 'httpcore._async.http_proxy', '/usr/local/lib/python3.10/site-packages/httpcore/_async/__init__.py': 'httpcore._async', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/mock.py': 'httpcore._backends.mock', '/usr/local/lib/python3.10/site-packages/httpcore/_backends/anyio.py': 'httpcore._backends.anyio', '/usr/local/lib/python3.10/site-packages/httpcore/__init__.py': 'httpcore', '/usr/local/lib/python3.10/site-packages/ollama/__init__.py': 'ollama', '/usr/local/lib/python3.10/site-packages/langchain_ollama/_compat.py': 'langchain_ollama._compat', '/usr/local/lib/python3.10/site-packages/langchain_ollama/_utils.py': 'langchain_ollama._utils', '/usr/local/lib/python3.10/site-packages/langchain_ollama/chat_models.py': 'langchain_ollama.chat_models', '/usr/local/lib/python3.10/site-packages/langchain_core/embeddings/__init__.py': 'langchain_core.embeddings', '/usr/local/lib/python3.10/site-packages/langchain_core/embeddings/embeddings.py': 'langchain_core.embeddings.embeddings', '/usr/local/lib/python3.10/site-packages/langchain_ollama/embeddings.py': 'langchain_ollama.embeddings', '/usr/local/lib/python3.10/site-packages/tenacity/_utils.py': 'tenacity._utils', '/usr/local/lib/python3.10/site-packages/tenacity/retry.py': 'tenacity.retry', '/usr/local/lib/python3.10/site-packages/tenacity/nap.py': 'tenacity.nap', '/usr/local/lib/python3.10/site-packages/tenacity/stop.py': 'tenacity.stop', '/usr/local/lib/python3.10/site-packages/tenacity/wait.py': 'tenacity.wait', '/usr/local/lib/python3.10/site-packages/tenacity/before.py': 'tenacity.before', '/usr/local/lib/python3.10/site-packages/tenacity/after.py': 'tenacity.after', '/usr/local/lib/python3.10/site-packages/tenacity/before_sleep.py': 'tenacity.before_sleep', '/usr/local/lib/python3.10/site-packages/tenacity/asyncio/retry.py': 'tenacity.asyncio.retry', '/usr/local/lib/python3.10/site-packages/tenacity/asyncio/__init__.py': 'tenacity.asyncio', '/usr/local/lib/python3.10/site-packages/tenacity/__init__.py': 'tenacity', '/usr/local/lib/python3.10/site-packages/langchain_core/language_models/llms.py': 'langchain_core.language_models.llms', '/usr/local/lib/python3.10/site-packages/langchain_ollama/llms.py': 'langchain_ollama.llms', '/usr/local/lib/python3.10/site-packages/langchain_ollama/__init__.py': 'langchain_ollama', '/workspace/pynguin/llm/deepmosa/model.py': 'pynguin.llm.deepmosa.model', '/workspace/pynguin/llm/deepmosa/llmseeding.py': 'pynguin.llm.deepmosa.llmseeding', '/workspace/pynguin/ga/algorithms/deepmosaalgorithm.py': 'pynguin.ga.algorithms.deepmosaalgorithm', '/workspace/pynguin/ga/algorithms/dynamosaalgorithm.py': 'pynguin.ga.algorithms.dynamosaalgorithm', '/workspace/pynguin/ga/algorithms/mioalgorithm.py': 'pynguin.ga.algorithms.mioalgorithm', '/workspace/pynguin/ga/algorithms/mosaalgorithm.py': 'pynguin.ga.algorithms.mosaalgorithm', '/workspace/pynguin/ga/algorithms/randomalgorithm.py': 'pynguin.ga.algorithms.randomalgorithm', '/workspace/pynguin/ga/algorithms/randomsearchalgorithm.py': 'pynguin.ga.algorithms.randomsearchalgorithm', '/workspace/pynguin/ga/algorithms/wholesuitealgorithm.py': 'pynguin.ga.algorithms.wholesuitealgorithm', '/workspace/pynguin/ga/operators/crossover.py': 'pynguin.ga.operators.crossover', '/workspace/pynguin/ga/operators/selection.py': 'pynguin.ga.operators.selection', '/workspace/pynguin/ga/stoppingcondition.py': 'pynguin.ga.stoppingcondition', '/workspace/pynguin/testcase/testfactory.py': 'pynguin.testcase.testfactory', '/workspace/pynguin/ga/chromosomefactory.py': 'pynguin.ga.chromosomefactory', '/workspace/pynguin/ga/testcasechromosomefactory.py': 'pynguin.ga.testcasechromosomefactory', '/workspace/pynguin/ga/testcasefactory.py': 'pynguin.ga.testcasefactory', '/workspace/pynguin/ga/testsuitechromosomefactory.py': 'pynguin.ga.testsuitechromosomefactory', '/workspace/pynguin/ga/generationalgorithmfactory.py': 'pynguin.ga.generationalgorithmfactory', '/workspace/pynguin/ga/postprocess.py': 'pynguin.ga.postprocess', '/usr/local/lib/python3.10/site-packages/astroid/typing.py': 'astroid.typing', '/usr/local/lib/python3.10/site-packages/astroid/exceptions.py': 'astroid.exceptions', '/usr/local/lib/python3.10/site-packages/astroid/util.py': 'astroid.util', '/usr/local/lib/python3.10/site-packages/astroid/context.py': 'astroid.context', '/usr/local/lib/python3.10/site-packages/astroid/decorators.py': 'astroid.decorators', '/usr/local/lib/python3.10/site-packages/astroid/const.py': 'astroid.const', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/__init__.py': 'astroid.interpreter', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/_import/__init__.py': 'astroid.interpreter._import', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/_import/util.py': 'astroid.interpreter._import.util', '/usr/local/lib/python3.10/site-packages/astroid/modutils.py': 'astroid.modutils', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/_import/spec.py': 'astroid.interpreter._import.spec', '/usr/local/lib/python3.10/site-packages/astroid/transforms.py': 'astroid.transforms', '/usr/local/lib/python3.10/site-packages/astroid/manager.py': 'astroid.manager', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/objectmodel.py': 'astroid.interpreter.objectmodel', '/usr/local/lib/python3.10/site-packages/astroid/bases.py': 'astroid.bases', '/usr/local/lib/python3.10/site-packages/astroid/protocols.py': 'astroid.protocols', '/usr/local/lib/python3.10/site-packages/astroid/interpreter/dunder_lookup.py': 'astroid.interpreter.dunder_lookup', '/usr/local/lib/python3.10/site-packages/astroid/nodes/as_string.py': 'astroid.nodes.as_string', '/usr/local/lib/python3.10/site-packages/astroid/nodes/const.py': 'astroid.nodes.const', '/usr/local/lib/python3.10/site-packages/astroid/nodes/utils.py': 'astroid.nodes.utils', '/usr/local/lib/python3.10/site-packages/astroid/nodes/node_ng.py': 'astroid.nodes.node_ng', '/usr/local/lib/python3.10/site-packages/astroid/nodes/_base_nodes.py': 'astroid.nodes._base_nodes', '/usr/local/lib/python3.10/site-packages/astroid/nodes/node_classes.py': 'astroid.nodes.node_classes', '/usr/local/lib/python3.10/site-packages/astroid/filter_statements.py': 'astroid.filter_statements', '/usr/local/lib/python3.10/site-packages/astroid/nodes/scoped_nodes/utils.py': 'astroid.nodes.scoped_nodes.utils', '/usr/local/lib/python3.10/site-packages/astroid/nodes/scoped_nodes/mixin.py': 'astroid.nodes.scoped_nodes.mixin', '/usr/local/lib/python3.10/site-packages/astroid/nodes/scoped_nodes/scoped_nodes.py': 'astroid.nodes.scoped_nodes.scoped_nodes', '/usr/local/lib/python3.10/site-packages/astroid/nodes/scoped_nodes/__init__.py': 'astroid.nodes.scoped_nodes', '/usr/local/lib/python3.10/site-packages/astroid/nodes/__init__.py': 'astroid.nodes', '/usr/local/lib/python3.10/site-packages/astroid/raw_building.py': 'astroid.raw_building', '/usr/local/lib/python3.10/site-packages/astroid/__pkginfo__.py': 'astroid.__pkginfo__', '/usr/local/lib/python3.10/site-packages/astroid/brain/__init__.py': 'astroid.brain', '/usr/local/lib/python3.10/site-packages/astroid/brain/helpers.py': 'astroid.brain.helpers', '/usr/local/lib/python3.10/site-packages/astroid/_ast.py': 'astroid._ast', '/usr/local/lib/python3.10/site-packages/astroid/rebuilder.py': 'astroid.rebuilder', '/usr/local/lib/python3.10/site-packages/astroid/builder.py': 'astroid.builder', '/usr/local/lib/python3.10/site-packages/astroid/inference_tip.py': 'astroid.inference_tip', '/usr/local/lib/python3.10/site-packages/astroid/objects.py': 'astroid.objects', '/usr/local/lib/python3.10/site-packages/astroid/arguments.py': 'astroid.arguments', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_argparse.py': 'astroid.brain.brain_argparse', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_attrs.py': 'astroid.brain.brain_attrs', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_boto3.py': 'astroid.brain.brain_boto3', '/usr/local/lib/python3.10/site-packages/astroid/helpers.py': 'astroid.helpers', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_builtin_inference.py': 'astroid.brain.brain_builtin_inference', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_collections.py': 'astroid.brain.brain_collections', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_crypt.py': 'astroid.brain.brain_crypt', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_ctypes.py': 'astroid.brain.brain_ctypes', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_curses.py': 'astroid.brain.brain_curses', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_dataclasses.py': 'astroid.brain.brain_dataclasses', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_datetime.py': 'astroid.brain.brain_datetime', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_dateutil.py': 'astroid.brain.brain_dateutil', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_functools.py': 'astroid.brain.brain_functools', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_gi.py': 'astroid.brain.brain_gi', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_hashlib.py': 'astroid.brain.brain_hashlib', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_http.py': 'astroid.brain.brain_http', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_hypothesis.py': 'astroid.brain.brain_hypothesis', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_io.py': 'astroid.brain.brain_io', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_mechanize.py': 'astroid.brain.brain_mechanize', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_multiprocessing.py': 'astroid.brain.brain_multiprocessing', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_namedtuple_enum.py': 'astroid.brain.brain_namedtuple_enum', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_nose.py': 'astroid.brain.brain_nose', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_einsumfunc.py': 'astroid.brain.brain_numpy_core_einsumfunc', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_fromnumeric.py': 'astroid.brain.brain_numpy_core_fromnumeric', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_utils.py': 'astroid.brain.brain_numpy_utils', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_function_base.py': 'astroid.brain.brain_numpy_core_function_base', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_multiarray.py': 'astroid.brain.brain_numpy_core_multiarray', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_numeric.py': 'astroid.brain.brain_numpy_core_numeric', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_numerictypes.py': 'astroid.brain.brain_numpy_core_numerictypes', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_core_umath.py': 'astroid.brain.brain_numpy_core_umath', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_ma.py': 'astroid.brain.brain_numpy_ma', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_ndarray.py': 'astroid.brain.brain_numpy_ndarray', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_numpy_random_mtrand.py': 'astroid.brain.brain_numpy_random_mtrand', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_pathlib.py': 'astroid.brain.brain_pathlib', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_pkg_resources.py': 'astroid.brain.brain_pkg_resources', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_pytest.py': 'astroid.brain.brain_pytest', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_qt.py': 'astroid.brain.brain_qt', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_random.py': 'astroid.brain.brain_random', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_re.py': 'astroid.brain.brain_re', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_regex.py': 'astroid.brain.brain_regex', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_responses.py': 'astroid.brain.brain_responses', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_scipy_signal.py': 'astroid.brain.brain_scipy_signal', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_signal.py': 'astroid.brain.brain_signal', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_six.py': 'astroid.brain.brain_six', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_sqlalchemy.py': 'astroid.brain.brain_sqlalchemy', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_ssl.py': 'astroid.brain.brain_ssl', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_subprocess.py': 'astroid.brain.brain_subprocess', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_threading.py': 'astroid.brain.brain_threading', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_type.py': 'astroid.brain.brain_type', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_typing.py': 'astroid.brain.brain_typing', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_unittest.py': 'astroid.brain.brain_unittest', '/usr/local/lib/python3.10/site-packages/astroid/brain/brain_uuid.py': 'astroid.brain.brain_uuid', '/usr/local/lib/python3.10/site-packages/astroid/astroid_manager.py': 'astroid.astroid_manager', '/usr/local/lib/python3.10/site-packages/astroid/__init__.py': 'astroid', '/workspace/pynguin/analyses/modulecomplexity.py': 'pynguin.analyses.modulecomplexity', '/workspace/pynguin/analyses/syntaxtree.py': 'pynguin.analyses.syntaxtree', '/workspace/pynguin/setup/testclustergenerator.py': 'pynguin.setup.testclustergenerator', '/workspace/pynguin/slicer/statementslicingobserver.py': 'pynguin.slicer.statementslicingobserver', '/workspace/pynguin/testcase/export.py': 'pynguin.testcase.export', '/workspace/pynguin/generator.py': 'pynguin.generator', '/workspace/project/isort/sorting.py': 'isort.sorting', '/workspace/project/isort/stdlibs/py27.py': 'isort.stdlibs.py27', '/workspace/project/isort/stdlibs/py2.py': 'isort.stdlibs.py2', '/workspace/project/isort/stdlibs/py36.py': 'isort.stdlibs.py36', '/workspace/project/isort/stdlibs/py37.py': 'isort.stdlibs.py37', '/workspace/project/isort/stdlibs/py38.py': 'isort.stdlibs.py38', '/workspace/project/isort/stdlibs/py39.py': 'isort.stdlibs.py39', '/workspace/project/isort/stdlibs/py310.py': 'isort.stdlibs.py310', '/workspace/project/isort/stdlibs/py311.py': 'isort.stdlibs.py311', '/workspace/project/isort/stdlibs/py312.py': 'isort.stdlibs.py312', '/workspace/project/isort/stdlibs/py313.py': 'isort.stdlibs.py313', '/workspace/project/isort/stdlibs/py314.py': 'isort.stdlibs.py314', '/workspace/project/isort/stdlibs/py3.py': 'isort.stdlibs.py3', '/workspace/project/isort/stdlibs/all.py': 'isort.stdlibs.all', '/workspace/project/isort/stdlibs/__init__.py': 'isort.stdlibs', '/workspace/project/isort/profiles.py': 'isort.profiles', '/workspace/project/isort/exceptions.py': 'isort.exceptions', '/workspace/project/isort/sections.py': 'isort.sections', '/workspace/project/isort/utils.py': 'isort.utils', '/workspace/project/isort/comments.py': 'isort.comments', '/workspace/project/isort/wrap_modes.py': 'isort.wrap_modes', '/workspace/project/isort/_vendored/tomli/_re.py': 'isort._vendored.tomli._re', '/workspace/project/isort/_vendored/tomli/_parser.py': 'isort._vendored.tomli._parser', '/workspace/project/isort/_vendored/tomli/__init__.py': 'isort._vendored.tomli', '/workspace/project/isort/settings.py': 'isort.settings', '/workspace/project/isort/_version.py': 'isort._version', '/workspace/project/isort/literal.py': 'isort.literal', '/workspace/project/isort/format.py': 'isort.format', '/workspace/project/isort/place.py': 'isort.place', '/workspace/project/isort/parse.py': 'isort.parse', '/workspace/project/isort/wrap.py': 'isort.wrap', '/workspace/project/isort/identify.py': 'isort.identify', '/workspace/project/isort/output.py': 'isort.output', '/workspace/project/isort/core.py': 'isort.core', '/workspace/project/isort/files.py': 'isort.files', '/workspace/project/isort/io.py': 'isort.io', '/workspace/project/isort/api.py': 'isort.api', '/workspace/project/isort/__init__.py': 'isort', '/usr/local/lib/python3.10/site-packages/astroid/constraint.py': 'astroid.constraint', '/usr/local/lib/python3.10/email/generator.py': 'email.generator', '/usr/local/lib/python3.10/ftplib.py': 'ftplib', '/usr/local/lib/python3.10/site-packages/langsmith/run_helpers.py': 'langsmith.run_helpers', '/usr/local/lib/python3.10/site-packages/langchain_core/env.py': 'langchain_core.env', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/_compat.py': 'langchain_core.tracers._compat', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/langchain.py': 'langchain_core.tracers.langchain', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/run_collector.py': 'langchain_core.tracers.run_collector', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/context.py': 'langchain_core.tracers.context', '/usr/local/lib/python3.10/site-packages/langchain_core/tracers/stdout.py': 'langchain_core.tracers.stdout', '/usr/local/lib/python3.10/site-packages/langchain_core/messages/block_translators/langchain_v0.py': 'langchain_core.messages.block_translators.langchain_v0', '/usr/local/lib/python3.10/site-packages/anyio/_backends/__init__.py': 'anyio._backends', '/usr/local/lib/python3.10/site-packages/anyio/_backends/_asyncio.py': 'anyio._backends._asyncio', '/usr/local/lib/python3.10/email/contentmanager.py': 'email.contentmanager', '/usr/local/lib/python3.10/email/policy.py': 'email.policy'}
    assert module_4.GEN_CREATED == 'GEN_CREATED'
    assert module_4.GEN_RUNNING == 'GEN_RUNNING'
    assert module_4.GEN_SUSPENDED == 'GEN_SUSPENDED'
    assert module_4.GEN_CLOSED == 'GEN_CLOSED'
    assert module_4.CORO_CREATED == 'CORO_CREATED'
    assert module_4.CORO_RUNNING == 'CORO_RUNNING'
    assert module_4.CORO_SUSPENDED == 'CORO_SUSPENDED'
    assert module_4.CORO_CLOSED == 'CORO_CLOSED'
    var_3 = module_0.check_code_string(var_0, var_1, var_0)
    assert var_3 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = module_0.sort_code_string(var_1, var_1, disregard_skip=var_1)
    assert var_4 == ''
    module_0.sort_code_string(var_4, config=var_0, file_path=var_0, disregard_skip=var_1, show_diff=var_3)

def test_case_15():
    var_0 = []
    var_1 = iter(var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.find_imports_in_paths(var_1, unique=var_2, top_only=var_2, **var_3)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_5 = list(var_4)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_5.thishost()
    assert module_5.MAXFTPCACHE == 10
    assert module_5.ftpcache == {}
    var_1 = module_0.find_imports_in_stream(var_0, _seen=var_0)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = {}
    var_3 = module_0.find_imports_in_paths(var_1, unique=var_0, **var_2)
    var_4 = list(var_3)
    module_0.check_code_string(var_4, var_2, var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '"Ekf-=d\x0bm1_,5'
    var_1 = None
    var_2 = module_0.find_imports_in_file(var_0, unique=var_1, top_only=var_1)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = {}
    var_4 = []
    var_5 = module_6.StringIO(*var_4, **var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = False
    var_7 = module_0.sort_stream(var_2, var_1, var_0, disregard_skip=var_1, show_diff=var_6)
    assert var_7 is False
    var_8 = var_5.__gt__(var_5)
    var_7.__reduce__()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_5.thishost()
    assert module_5.MAXFTPCACHE == 10
    assert module_5.ftpcache == {}
    var_1 = None
    var_2 = module_0.find_imports_in_stream(var_0, _seen=var_1)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = True
    var_4 = {}
    var_5 = module_0.find_imports_in_file(var_4, var_0, var_4, var_4)
    var_6 = module_0.find_imports_in_paths(var_2, unique=var_3, **var_4)
    var_7 = list(var_6)
    var_8 = None
    var_0.find_user_password(var_8, var_2)

def test_case_19():
    var_0 = module_5.thishost()
    assert module_5.MAXFTPCACHE == 10
    assert module_5.ftpcache == {}
    var_1 = 'test_dir'
    var_2 = module_0.check_stream(var_1)
    assert var_2 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_3 = True
    var_4 = {}
    var_5 = module_0.find_imports_in_file(var_4, var_0, var_4, var_4)
    var_6 = module_0.find_imports_in_paths(var_5, unique=var_3, **var_4)
    with pytest.raises(TypeError):
        var_7 = list(var_6)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_5.thishost()
    assert module_5.MAXFTPCACHE == 10
    assert module_5.ftpcache == {}
    var_1 = module_0.check_stream(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = module_0.find_imports_in_stream(var_0, _seen=var_0)
    var_3 = {}
    var_4 = module_0.find_imports_in_paths(var_2, unique=var_0, **var_3)
    var_5 = list(var_4)
    var_6 = None
    var_7 = module_0.sort_code_string(var_6, var_6, show_diff=var_0)
    assert var_7 == ''
    module_7.indent(var_5, var_5)

def test_case_21():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_6.StringIO(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = module_8.in_table_c11(var_2)
    assert var_4 is False
    assert f'{type(module_8.unicodedata).__module__}.{type(module_8.unicodedata).__qualname__}' == 'unicodedata.UCD'
    assert module_8.b1_set == {65024, 65025, 65026, 65027, 65028, 65029, 6150, 65030, 65031, 65032, 65033, 6155, 6156, 6157, 8203, 8204, 8205, 65034, 65035, 65036, 65037, 65038, 65039, 173, 847, 8288, 65279}
    assert module_8.b3_exceptions == {181: 'μ', 223: 'ss', 304: 'i̇', 329: 'ʼn', 383: 's', 496: 'ǰ', 837: 'ι', 890: ' ι', 912: 'ΐ', 944: 'ΰ', 962: 'σ', 976: 'β', 977: 'θ', 978: 'υ', 979: 'ύ', 980: 'ϋ', 981: 'φ', 982: 'π', 1008: 'κ', 1009: 'ρ', 1010: 'σ', 1013: 'ε', 1415: 'եւ', 7830: 'ẖ', 7831: 'ẗ', 7832: 'ẘ', 7833: 'ẙ', 7834: 'aʾ', 7835: 'ṡ', 8016: 'ὐ', 8018: 'ὒ', 8020: 'ὔ', 8022: 'ὖ', 8064: 'ἀι', 8065: 'ἁι', 8066: 'ἂι', 8067: 'ἃι', 8068: 'ἄι', 8069: 'ἅι', 8070: 'ἆι', 8071: 'ἇι', 8072: 'ἀι', 8073: 'ἁι', 8074: 'ἂι', 8075: 'ἃι', 8076: 'ἄι', 8077: 'ἅι', 8078: 'ἆι', 8079: 'ἇι', 8080: 'ἠι', 8081: 'ἡι', 8082: 'ἢι', 8083: 'ἣι', 8084: 'ἤι', 8085: 'ἥι', 8086: 'ἦι', 8087: 'ἧι', 8088: 'ἠι', 8089: 'ἡι', 8090: 'ἢι', 8091: 'ἣι', 8092: 'ἤι', 8093: 'ἥι', 8094: 'ἦι', 8095: 'ἧι', 8096: 'ὠι', 8097: 'ὡι', 8098: 'ὢι', 8099: 'ὣι', 8100: 'ὤι', 8101: 'ὥι', 8102: 'ὦι', 8103: 'ὧι', 8104: 'ὠι', 8105: 'ὡι', 8106: 'ὢι', 8107: 'ὣι', 8108: 'ὤι', 8109: 'ὥι', 8110: 'ὦι', 8111: 'ὧι', 8114: 'ὰι', 8115: 'αι', 8116: 'άι', 8118: 'ᾶ', 8119: 'ᾶι', 8124: 'αι', 8126: 'ι', 8130: 'ὴι', 8131: 'ηι', 8132: 'ήι', 8134: 'ῆ', 8135: 'ῆι', 8140: 'ηι', 8146: 'ῒ', 8147: 'ΐ', 8150: 'ῖ', 8151: 'ῗ', 8162: 'ῢ', 8163: 'ΰ', 8164: 'ῤ', 8166: 'ῦ', 8167: 'ῧ', 8178: 'ὼι', 8179: 'ωι', 8180: 'ώι', 8182: 'ῶ', 8183: 'ῶι', 8188: 'ωι', 8360: 'rs', 8450: 'c', 8451: '°c', 8455: 'ɛ', 8457: '°f', 8459: 'h', 8460: 'h', 8461: 'h', 8464: 'i', 8465: 'i', 8466: 'l', 8469: 'n', 8470: 'no', 8473: 'p', 8474: 'q', 8475: 'r', 8476: 'r', 8477: 'r', 8480: 'sm', 8481: 'tel', 8482: 'tm', 8484: 'z', 8488: 'z', 8492: 'b', 8493: 'c', 8496: 'e', 8497: 'f', 8499: 'm', 8510: 'γ', 8511: 'π', 8517: 'd', 13169: 'hpa', 13171: 'au', 13173: 'ov', 13184: 'pa', 13185: 'na', 13186: 'μa', 13187: 'ma', 13188: 'ka', 13189: 'kb', 13190: 'mb', 13191: 'gb', 13194: 'pf', 13195: 'nf', 13196: 'μf', 13200: 'hz', 13201: 'khz', 13202: 'mhz', 13203: 'ghz', 13204: 'thz', 13225: 'pa', 13226: 'kpa', 13227: 'mpa', 13228: 'gpa', 13236: 'pv', 13237: 'nv', 13238: 'μv', 13239: 'mv', 13240: 'kv', 13241: 'mv', 13242: 'pw', 13243: 'nw', 13244: 'μw', 13245: 'mw', 13246: 'kw', 13247: 'mw', 13248: 'kω', 13249: 'mω', 13251: 'bq', 13254: 'c∕kg', 13255: 'co.', 13256: 'db', 13257: 'gy', 13259: 'hp', 13261: 'kk', 13262: 'km', 13271: 'ph', 13273: 'ppm', 13274: 'pr', 13276: 'sv', 13277: 'wb', 64256: 'ff', 64257: 'fi', 64258: 'fl', 64259: 'ffi', 64260: 'ffl', 64261: 'st', 64262: 'st', 64275: 'մն', 64276: 'մե', 64277: 'մի', 64278: 'վն', 64279: 'մխ', 119808: 'a', 119809: 'b', 119810: 'c', 119811: 'd', 119812: 'e', 119813: 'f', 119814: 'g', 119815: 'h', 119816: 'i', 119817: 'j', 119818: 'k', 119819: 'l', 119820: 'm', 119821: 'n', 119822: 'o', 119823: 'p', 119824: 'q', 119825: 'r', 119826: 's', 119827: 't', 119828: 'u', 119829: 'v', 119830: 'w', 119831: 'x', 119832: 'y', 119833: 'z', 119860: 'a', 119861: 'b', 119862: 'c', 119863: 'd', 119864: 'e', 119865: 'f', 119866: 'g', 119867: 'h', 119868: 'i', 119869: 'j', 119870: 'k', 119871: 'l', 119872: 'm', 119873: 'n', 119874: 'o', 119875: 'p', 119876: 'q', 119877: 'r', 119878: 's', 119879: 't', 119880: 'u', 119881: 'v', 119882: 'w', 119883: 'x', 119884: 'y', 119885: 'z', 119912: 'a', 119913: 'b', 119914: 'c', 119915: 'd', 119916: 'e', 119917: 'f', 119918: 'g', 119919: 'h', 119920: 'i', 119921: 'j', 119922: 'k', 119923: 'l', 119924: 'm', 119925: 'n', 119926: 'o', 119927: 'p', 119928: 'q', 119929: 'r', 119930: 's', 119931: 't', 119932: 'u', 119933: 'v', 119934: 'w', 119935: 'x', 119936: 'y', 119937: 'z', 119964: 'a', 119966: 'c', 119967: 'd', 119970: 'g', 119973: 'j', 119974: 'k', 119977: 'n', 119978: 'o', 119979: 'p', 119980: 'q', 119982: 's', 119983: 't', 119984: 'u', 119985: 'v', 119986: 'w', 119987: 'x', 119988: 'y', 119989: 'z', 120016: 'a', 120017: 'b', 120018: 'c', 120019: 'd', 120020: 'e', 120021: 'f', 120022: 'g', 120023: 'h', 120024: 'i', 120025: 'j', 120026: 'k', 120027: 'l', 120028: 'm', 120029: 'n', 120030: 'o', 120031: 'p', 120032: 'q', 120033: 'r', 120034: 's', 120035: 't', 120036: 'u', 120037: 'v', 120038: 'w', 120039: 'x', 120040: 'y', 120041: 'z', 120068: 'a', 120069: 'b', 120071: 'd', 120072: 'e', 120073: 'f', 120074: 'g', 120077: 'j', 120078: 'k', 120079: 'l', 120080: 'm', 120081: 'n', 120082: 'o', 120083: 'p', 120084: 'q', 120086: 's', 120087: 't', 120088: 'u', 120089: 'v', 120090: 'w', 120091: 'x', 120092: 'y', 120120: 'a', 120121: 'b', 120123: 'd', 120124: 'e', 120125: 'f', 120126: 'g', 120128: 'i', 120129: 'j', 120130: 'k', 120131: 'l', 120132: 'm', 120134: 'o', 120138: 's', 120139: 't', 120140: 'u', 120141: 'v', 120142: 'w', 120143: 'x', 120144: 'y', 120172: 'a', 120173: 'b', 120174: 'c', 120175: 'd', 120176: 'e', 120177: 'f', 120178: 'g', 120179: 'h', 120180: 'i', 120181: 'j', 120182: 'k', 120183: 'l', 120184: 'm', 120185: 'n', 120186: 'o', 120187: 'p', 120188: 'q', 120189: 'r', 120190: 's', 120191: 't', 120192: 'u', 120193: 'v', 120194: 'w', 120195: 'x', 120196: 'y', 120197: 'z', 120224: 'a', 120225: 'b', 120226: 'c', 120227: 'd', 120228: 'e', 120229: 'f', 120230: 'g', 120231: 'h', 120232: 'i', 120233: 'j', 120234: 'k', 120235: 'l', 120236: 'm', 120237: 'n', 120238: 'o', 120239: 'p', 120240: 'q', 120241: 'r', 120242: 's', 120243: 't', 120244: 'u', 120245: 'v', 120246: 'w', 120247: 'x', 120248: 'y', 120249: 'z', 120276: 'a', 120277: 'b', 120278: 'c', 120279: 'd', 120280: 'e', 120281: 'f', 120282: 'g', 120283: 'h', 120284: 'i', 120285: 'j', 120286: 'k', 120287: 'l', 120288: 'm', 120289: 'n', 120290: 'o', 120291: 'p', 120292: 'q', 120293: 'r', 120294: 's', 120295: 't', 120296: 'u', 120297: 'v', 120298: 'w', 120299: 'x', 120300: 'y', 120301: 'z', 120328: 'a', 120329: 'b', 120330: 'c', 120331: 'd', 120332: 'e', 120333: 'f', 120334: 'g', 120335: 'h', 120336: 'i', 120337: 'j', 120338: 'k', 120339: 'l', 120340: 'm', 120341: 'n', 120342: 'o', 120343: 'p', 120344: 'q', 120345: 'r', 120346: 's', 120347: 't', 120348: 'u', 120349: 'v', 120350: 'w', 120351: 'x', 120352: 'y', 120353: 'z', 120380: 'a', 120381: 'b', 120382: 'c', 120383: 'd', 120384: 'e', 120385: 'f', 120386: 'g', 120387: 'h', 120388: 'i', 120389: 'j', 120390: 'k', 120391: 'l', 120392: 'm', 120393: 'n', 120394: 'o', 120395: 'p', 120396: 'q', 120397: 'r', 120398: 's', 120399: 't', 120400: 'u', 120401: 'v', 120402: 'w', 120403: 'x', 120404: 'y', 120405: 'z', 120432: 'a', 120433: 'b', 120434: 'c', 120435: 'd', 120436: 'e', 120437: 'f', 120438: 'g', 120439: 'h', 120440: 'i', 120441: 'j', 120442: 'k', 120443: 'l', 120444: 'm', 120445: 'n', 120446: 'o', 120447: 'p', 120448: 'q', 120449: 'r', 120450: 's', 120451: 't', 120452: 'u', 120453: 'v', 120454: 'w', 120455: 'x', 120456: 'y', 120457: 'z', 120488: 'α', 120489: 'β', 120490: 'γ', 120491: 'δ', 120492: 'ε', 120493: 'ζ', 120494: 'η', 120495: 'θ', 120496: 'ι', 120497: 'κ', 120498: 'λ', 120499: 'μ', 120500: 'ν', 120501: 'ξ', 120502: 'ο', 120503: 'π', 120504: 'ρ', 120505: 'θ', 120506: 'σ', 120507: 'τ', 120508: 'υ', 120509: 'φ', 120510: 'χ', 120511: 'ψ', 120512: 'ω', 120531: 'σ', 120546: 'α', 120547: 'β', 120548: 'γ', 120549: 'δ', 120550: 'ε', 120551: 'ζ', 120552: 'η', 120553: 'θ', 120554: 'ι', 120555: 'κ', 120556: 'λ', 120557: 'μ', 120558: 'ν', 120559: 'ξ', 120560: 'ο', 120561: 'π', 120562: 'ρ', 120563: 'θ', 120564: 'σ', 120565: 'τ', 120566: 'υ', 120567: 'φ', 120568: 'χ', 120569: 'ψ', 120570: 'ω', 120589: 'σ', 120604: 'α', 120605: 'β', 120606: 'γ', 120607: 'δ', 120608: 'ε', 120609: 'ζ', 120610: 'η', 120611: 'θ', 120612: 'ι', 120613: 'κ', 120614: 'λ', 120615: 'μ', 120616: 'ν', 120617: 'ξ', 120618: 'ο', 120619: 'π', 120620: 'ρ', 120621: 'θ', 120622: 'σ', 120623: 'τ', 120624: 'υ', 120625: 'φ', 120626: 'χ', 120627: 'ψ', 120628: 'ω', 120647: 'σ', 120662: 'α', 120663: 'β', 120664: 'γ', 120665: 'δ', 120666: 'ε', 120667: 'ζ', 120668: 'η', 120669: 'θ', 120670: 'ι', 120671: 'κ', 120672: 'λ', 120673: 'μ', 120674: 'ν', 120675: 'ξ', 120676: 'ο', 120677: 'π', 120678: 'ρ', 120679: 'θ', 120680: 'σ', 120681: 'τ', 120682: 'υ', 120683: 'φ', 120684: 'χ', 120685: 'ψ', 120686: 'ω', 120705: 'σ', 120720: 'α', 120721: 'β', 120722: 'γ', 120723: 'δ', 120724: 'ε', 120725: 'ζ', 120726: 'η', 120727: 'θ', 120728: 'ι', 120729: 'κ', 120730: 'λ', 120731: 'μ', 120732: 'ν', 120733: 'ξ', 120734: 'ο', 120735: 'π', 120736: 'ρ', 120737: 'θ', 120738: 'σ', 120739: 'τ', 120740: 'υ', 120741: 'φ', 120742: 'χ', 120743: 'ψ', 120744: 'ω', 120763: 'σ'}
    assert module_8.c22_specials == {8204, 8205, 6158, 1807, 8232, 8233, 1757, 65529, 8288, 8289, 8290, 8291, 65530, 8298, 8299, 8300, 8301, 8302, 8303, 119155, 119156, 119157, 119158, 119159, 119160, 119161, 119162, 65531, 65532, 65279}
    assert module_8.c6_set == {65529, 65530, 65531, 65532, 65533}
    assert module_8.c7_set == {12272, 12273, 12274, 12275, 12276, 12277, 12278, 12279, 12280, 12281, 12282, 12283}
    assert module_8.c8_set == {832, 833, 8234, 8235, 8236, 8237, 8206, 8207, 8238, 8298, 8299, 8300, 8301, 8302, 8303}
    assert module_8.c9_set == {917505, 917536, 917537, 917538, 917539, 917540, 917541, 917542, 917543, 917544, 917545, 917546, 917547, 917548, 917549, 917550, 917551, 917552, 917553, 917554, 917555, 917556, 917557, 917558, 917559, 917560, 917561, 917562, 917563, 917564, 917565, 917566, 917567, 917568, 917569, 917570, 917571, 917572, 917573, 917574, 917575, 917576, 917577, 917578, 917579, 917580, 917581, 917582, 917583, 917584, 917585, 917586, 917587, 917588, 917589, 917590, 917591, 917592, 917593, 917594, 917595, 917596, 917597, 917598, 917599, 917600, 917601, 917602, 917603, 917604, 917605, 917606, 917607, 917608, 917609, 917610, 917611, 917612, 917613, 917614, 917615, 917616, 917617, 917618, 917619, 917620, 917621, 917622, 917623, 917624, 917625, 917626, 917627, 917628, 917629, 917630, 917631}
    var_5 = 'py'
    var_6 = {}
    var_7 = module_0.check_stream(var_3, extension=var_5, **var_6)
    assert var_7 is False
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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

def test_case_22():
    var_0 = 'import sys\nif True:'
    var_1 = []
    var_2 = {}
    var_3 = module_6.StringIO(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_4 = [var_0]
    var_5 = module_6.StringIO(*var_4, **var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == '_io.StringIO'
    var_6 = True
    var_7 = 'atomic'
    var_8 = {var_7: var_6}
    var_9 = module_2.Config(**var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.settings.Config'
    assert var_9.py_version == 'py3'
    assert f'{type(var_9.force_to_top).__module__}.{type(var_9.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.force_to_top) == 0
    assert f'{type(var_9.skip).__module__}.{type(var_9.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip) == 19
    assert f'{type(var_9.extend_skip).__module__}.{type(var_9.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip) == 0
    assert f'{type(var_9.skip_glob).__module__}.{type(var_9.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip_glob) == 0
    assert f'{type(var_9.extend_skip_glob).__module__}.{type(var_9.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip_glob) == 0
    assert var_9.skip_gitignore is False
    assert var_9.line_length == 79
    assert var_9.wrap_length == 0
    assert var_9.line_ending == ''
    assert var_9.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_9.no_sections is False
    assert f'{type(var_9.known_future_library).__module__}.{type(var_9.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_future_library) == 1
    assert f'{type(var_9.known_third_party).__module__}.{type(var_9.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_third_party) == 0
    assert f'{type(var_9.known_first_party).__module__}.{type(var_9.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_first_party) == 0
    assert f'{type(var_9.known_local_folder).__module__}.{type(var_9.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_local_folder) == 0
    assert f'{type(var_9.known_standard_library).__module__}.{type(var_9.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_standard_library) == 234
    assert f'{type(var_9.extra_standard_library).__module__}.{type(var_9.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extra_standard_library) == 0
    assert var_9.known_other == {}
    assert var_9.multi_line_output == module_1.WrapModes.GRID
    assert var_9.forced_separate == ()
    assert var_9.indent == '    '
    assert var_9.comment_prefix == '  #'
    assert var_9.length_sort is False
    assert var_9.length_sort_straight is False
    assert f'{type(var_9.length_sort_sections).__module__}.{type(var_9.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.length_sort_sections) == 0
    assert f'{type(var_9.add_imports).__module__}.{type(var_9.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.add_imports) == 0
    assert f'{type(var_9.remove_imports).__module__}.{type(var_9.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.remove_imports) == 0
    assert var_9.append_only is False
    assert var_9.reverse_relative is False
    assert var_9.force_single_line is False
    assert var_9.single_line_exclusions == ()
    assert var_9.default_section == 'THIRDPARTY'
    assert var_9.import_headings == {}
    assert var_9.import_footers == {}
    assert var_9.balanced_wrapping is False
    assert var_9.use_parentheses is False
    assert var_9.order_by_type is True
    assert var_9.atomic is True
    assert var_9.lines_before_imports == -1
    assert var_9.lines_after_imports == -1
    assert var_9.lines_between_sections == 1
    assert var_9.lines_between_types == 0
    assert var_9.combine_as_imports is False
    assert var_9.combine_star is False
    assert var_9.include_trailing_comma is False
    assert var_9.from_first is False
    assert var_9.verbose is False
    assert var_9.quiet is False
    assert var_9.force_adds is False
    assert var_9.force_alphabetical_sort_within_sections is False
    assert var_9.force_alphabetical_sort is False
    assert var_9.force_grid_wrap == 0
    assert var_9.force_sort_within_sections is False
    assert var_9.lexicographical is False
    assert var_9.group_by_package is False
    assert var_9.ignore_whitespace is False
    assert f'{type(var_9.no_lines_before).__module__}.{type(var_9.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.no_lines_before) == 0
    assert var_9.no_inline_sort is False
    assert var_9.ignore_comments is False
    assert var_9.case_sensitive is False
    assert f'{type(var_9.sources).__module__}.{type(var_9.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_9.sources) == 2
    assert var_9.virtual_env == ''
    assert var_9.conda_env == ''
    assert var_9.ensure_newline_before_comments is False
    assert var_9.directory == '/workspace/run'
    assert var_9.profile == ''
    assert var_9.honor_noqa is False
    assert f'{type(var_9.src_paths).__module__}.{type(var_9.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_9.src_paths) == 2
    assert var_9.remove_redundant_aliases is False
    assert var_9.float_to_top is False
    assert var_9.filter_files is False
    assert var_9.formatter == ''
    assert var_9.formatting_function is None
    assert var_9.color_output is False
    assert f'{type(var_9.treat_comments_as_code).__module__}.{type(var_9.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.treat_comments_as_code) == 0
    assert var_9.treat_all_comments_as_code is False
    assert f'{type(var_9.supported_extensions).__module__}.{type(var_9.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.supported_extensions) == 4
    assert f'{type(var_9.blocked_extensions).__module__}.{type(var_9.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.blocked_extensions) == 1
    assert f'{type(var_9.constants).__module__}.{type(var_9.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.constants) == 0
    assert f'{type(var_9.classes).__module__}.{type(var_9.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.classes) == 0
    assert f'{type(var_9.variables).__module__}.{type(var_9.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.variables) == 0
    assert var_9.dedup_headings is False
    assert var_9.only_sections is False
    assert var_9.only_modified is False
    assert var_9.combine_straight_imports is False
    assert var_9.auto_identify_namespace_packages is True
    assert f'{type(var_9.namespace_packages).__module__}.{type(var_9.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.namespace_packages) == 0
    assert var_9.follow_links is True
    assert var_9.indented_import_headings is True
    assert var_9.honor_case_in_force_sorted_sections is False
    assert var_9.sort_relative_in_force_sorted_sections is False
    assert var_9.overwrite_in_place is False
    assert var_9.reverse_sort is False
    assert var_9.star_first is False
    assert var_9.git_ls_files == {}
    assert var_9.format_error == '{error}: {message}'
    assert var_9.format_success == '{success}: {message}'
    assert var_9.sort_order == 'natural'
    assert var_9.sort_reexports is False
    assert var_9.split_on_trailing_comma is False
    assert module_2.TYPE_CHECKING is False
    assert module_2.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_2.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.FIRSTPARTY == 'FIRSTPARTY'
    assert module_2.FUTURE == 'FUTURE'
    assert module_2.LOCALFOLDER == 'LOCALFOLDER'
    assert module_2.STDLIB == 'STDLIB'
    assert module_2.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_2.CYTHON_EXTENSIONS).__module__}.{type(module_2.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_2.SUPPORTED_EXTENSIONS).__module__}.{type(module_2.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_2.BLOCKED_EXTENSIONS).__module__}.{type(module_2.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.BLOCKED_EXTENSIONS) == 1
    assert module_2.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_2.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_2.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_2.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_2.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_2.DEFAULT_SKIP).__module__}.{type(module_2.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_SKIP) == 19
    assert module_2.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_2.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_2.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_2.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_2.KNOWN_PREFIX == 'known_'
    assert module_2.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_2.RUNTIME_SOURCE == 'runtime'
    assert module_2.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_2.DEFAULT_CONFIG).__module__}.{type(module_2.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_2.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_2.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_2.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip).__module__}.{type(module_2.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_2.DEFAULT_CONFIG.skip_gitignore is False
    assert module_2.DEFAULT_CONFIG.line_length == 79
    assert module_2.DEFAULT_CONFIG.wrap_length == 0
    assert module_2.DEFAULT_CONFIG.line_ending == ''
    assert module_2.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_2.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_2.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_2.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_2.DEFAULT_CONFIG.known_other == {}
    assert module_2.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_2.DEFAULT_CONFIG.forced_separate == ()
    assert module_2.DEFAULT_CONFIG.indent == '    '
    assert module_2.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_2.DEFAULT_CONFIG.length_sort is False
    assert module_2.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_2.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_2.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.add_imports).__module__}.{type(module_2.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_2.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.remove_imports) == 0
    assert module_2.DEFAULT_CONFIG.append_only is False
    assert module_2.DEFAULT_CONFIG.reverse_relative is False
    assert module_2.DEFAULT_CONFIG.force_single_line is False
    assert module_2.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_2.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_2.DEFAULT_CONFIG.import_headings == {}
    assert module_2.DEFAULT_CONFIG.import_footers == {}
    assert module_2.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_2.DEFAULT_CONFIG.use_parentheses is False
    assert module_2.DEFAULT_CONFIG.order_by_type is True
    assert module_2.DEFAULT_CONFIG.atomic is False
    assert module_2.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_2.DEFAULT_CONFIG.lines_between_types == 0
    assert module_2.DEFAULT_CONFIG.combine_as_imports is False
    assert module_2.DEFAULT_CONFIG.combine_star is False
    assert module_2.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_2.DEFAULT_CONFIG.from_first is False
    assert module_2.DEFAULT_CONFIG.verbose is False
    assert module_2.DEFAULT_CONFIG.quiet is False
    assert module_2.DEFAULT_CONFIG.force_adds is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_2.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_2.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.lexicographical is False
    assert module_2.DEFAULT_CONFIG.group_by_package is False
    assert module_2.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_2.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_2.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_2.DEFAULT_CONFIG.no_inline_sort is False
    assert module_2.DEFAULT_CONFIG.ignore_comments is False
    assert module_2.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_2.DEFAULT_CONFIG.sources).__module__}.{type(module_2.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.sources) == 1
    assert module_2.DEFAULT_CONFIG.virtual_env == ''
    assert module_2.DEFAULT_CONFIG.conda_env == ''
    assert module_2.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_2.DEFAULT_CONFIG.directory == '/workspace/run'
    assert module_2.DEFAULT_CONFIG.profile == ''
    assert module_2.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_2.DEFAULT_CONFIG.src_paths).__module__}.{type(module_2.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.src_paths) == 2
    assert module_2.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_2.DEFAULT_CONFIG.float_to_top is False
    assert module_2.DEFAULT_CONFIG.filter_files is False
    assert module_2.DEFAULT_CONFIG.formatter == ''
    assert module_2.DEFAULT_CONFIG.formatting_function is None
    assert module_2.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_2.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_2.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_2.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.constants).__module__}.{type(module_2.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.classes).__module__}.{type(module_2.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.variables).__module__}.{type(module_2.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.variables) == 0
    assert module_2.DEFAULT_CONFIG.dedup_headings is False
    assert module_2.DEFAULT_CONFIG.only_sections is False
    assert module_2.DEFAULT_CONFIG.only_modified is False
    assert module_2.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_2.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_2.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_2.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_2.DEFAULT_CONFIG.follow_links is True
    assert module_2.DEFAULT_CONFIG.indented_import_headings is True
    assert module_2.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_2.DEFAULT_CONFIG.reverse_sort is False
    assert module_2.DEFAULT_CONFIG.star_first is False
    assert module_2.DEFAULT_CONFIG.git_ls_files == {}
    assert module_2.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_2.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_2.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_2.DEFAULT_CONFIG.sort_reexports is False
    assert module_2.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_2.Config.known_patterns).__module__}.{type(module_2.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments).__module__}.{type(module_2.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments_end).__module__}.{type(module_2.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skips).__module__}.{type(module_2.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skip_globs).__module__}.{type(module_2.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.sorting_function).__module__}.{type(module_2.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_10 = {}
    with pytest.raises(module_9.ExistingSyntaxErrors):
        module_0.sort_stream(var_5, var_3, config=var_9, **var_10)

def test_case_23():
    var_0 = module_5.thishost()
    assert module_5.MAXFTPCACHE == 10
    assert module_5.ftpcache == {}
    var_1 = module_0.check_stream(var_0)
    assert var_1 is True
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_2 = module_0.find_imports_in_stream(var_0, _seen=var_0)
    var_3 = {}
    var_4 = var_0.__eq__(var_3)
    var_5 = module_0.find_imports_in_code(var_0, unique=var_0)
    with pytest.raises(TypeError):
        var_6 = list(var_5)

def test_case_24():
    var_0 = {}
    var_1 = module_10.LocalPart(**var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'email._header_value_parser.LocalPart'
    assert len(var_1) == 0
    assert module_10.hexdigits == '0123456789abcdefABCDEF'
    assert module_10.WSP == {'\t', ' '}
    assert module_10.CFWS_LEADER == {'\t', '(', ' '}
    assert module_10.SPECIALS == {'"', '\\', '>', ':', '(', '.', '[', '<', ';', '@', ',', ']', ')'}
    assert module_10.ATOM_ENDS == {'"', '\\', '>', ':', '(', '.', '[', '<', ' ', '\t', ';', '@', ',', ']', ')'}
    assert module_10.DOT_ATOM_ENDS == {'"', '\\', '>', ':', '(', '[', '<', ' ', '\t', ';', '@', ',', ']', ')'}
    assert module_10.PHRASE_ENDS == {'\\', '>', ':', '[', '<', ';', '@', ',', ']', ')'}
    assert module_10.TSPECIALS == {'"', '\\', '=', '>', ':', '(', '?', '[', '<', ';', '@', ',', '/', ']', ')'}
    assert module_10.TOKEN_ENDS == {'"', '\\', '=', '>', ':', '(', '?', '[', '<', ' ', '\t', ';', '@', ',', '/', ']', ')'}
    assert module_10.ASPECIALS == {'"', '\\', '=', '>', ':', '(', '?', '%', '[', '<', ';', '@', ',', '/', "'", '*', ']', ')'}
    assert module_10.ATTRIBUTE_ENDS == {'<', ';', '@', "'", ')', '(', '?', '%', ' ', ',', '/', '"', '\\', '=', '>', '[', ':', '\t', '*', ']'}
    assert module_10.EXTENDED_ATTRIBUTE_ENDS == {'<', ';', '@', "'", ')', '(', '?', ' ', ',', '/', '"', '\\', '=', '>', '[', ':', '\t', '*', ']'}
    assert module_10.NLSET == {'\r', '\n'}
    assert module_10.SPECIALSNL == {'\r', '"', '\\', ':', '>', '(', '.', '[', '<', ';', '@', ',', ']', '\n', ')'}
    assert f'{type(module_10.rfc2047_matcher).__module__}.{type(module_10.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_10.DOT).__module__}.{type(module_10.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_10.DOT) == 1
    assert f'{type(module_10.ListSeparator).__module__}.{type(module_10.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_10.ListSeparator) == 1
    assert f'{type(module_10.RouteComponentMarker).__module__}.{type(module_10.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_10.RouteComponentMarker) == 1
    assert module_10.LocalPart.token_type == 'local-part'
    assert module_10.LocalPart.as_ew_allowed is False
    assert f'{type(module_10.LocalPart.value).__module__}.{type(module_10.LocalPart.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_10.LocalPart.local_part).__module__}.{type(module_10.LocalPart.local_part).__qualname__}' == 'builtins.property'
    var_2 = None
    var_3 = module_0.find_imports_in_stream(var_2, file_path=var_2)
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_4 = {}
    var_5 = module_6.StringIO(*var_1, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_6 = True
    var_7 = 'atomic'
    var_8 = {var_7: var_6}
    var_9 = module_2.Config(**var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.settings.Config'
    assert var_9.py_version == 'py3'
    assert f'{type(var_9.force_to_top).__module__}.{type(var_9.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.force_to_top) == 0
    assert f'{type(var_9.skip).__module__}.{type(var_9.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip) == 19
    assert f'{type(var_9.extend_skip).__module__}.{type(var_9.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip) == 0
    assert f'{type(var_9.skip_glob).__module__}.{type(var_9.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.skip_glob) == 0
    assert f'{type(var_9.extend_skip_glob).__module__}.{type(var_9.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extend_skip_glob) == 0
    assert var_9.skip_gitignore is False
    assert var_9.line_length == 79
    assert var_9.wrap_length == 0
    assert var_9.line_ending == ''
    assert var_9.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_9.no_sections is False
    assert f'{type(var_9.known_future_library).__module__}.{type(var_9.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_future_library) == 1
    assert f'{type(var_9.known_third_party).__module__}.{type(var_9.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_third_party) == 0
    assert f'{type(var_9.known_first_party).__module__}.{type(var_9.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_first_party) == 0
    assert f'{type(var_9.known_local_folder).__module__}.{type(var_9.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_local_folder) == 0
    assert f'{type(var_9.known_standard_library).__module__}.{type(var_9.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.known_standard_library) == 234
    assert f'{type(var_9.extra_standard_library).__module__}.{type(var_9.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.extra_standard_library) == 0
    assert var_9.known_other == {}
    assert var_9.multi_line_output == module_1.WrapModes.GRID
    assert var_9.forced_separate == ()
    assert var_9.indent == '    '
    assert var_9.comment_prefix == '  #'
    assert var_9.length_sort is False
    assert var_9.length_sort_straight is False
    assert f'{type(var_9.length_sort_sections).__module__}.{type(var_9.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.length_sort_sections) == 0
    assert f'{type(var_9.add_imports).__module__}.{type(var_9.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.add_imports) == 0
    assert f'{type(var_9.remove_imports).__module__}.{type(var_9.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.remove_imports) == 0
    assert var_9.append_only is False
    assert var_9.reverse_relative is False
    assert var_9.force_single_line is False
    assert var_9.single_line_exclusions == ()
    assert var_9.default_section == 'THIRDPARTY'
    assert var_9.import_headings == {}
    assert var_9.import_footers == {}
    assert var_9.balanced_wrapping is False
    assert var_9.use_parentheses is False
    assert var_9.order_by_type is True
    assert var_9.atomic is True
    assert var_9.lines_before_imports == -1
    assert var_9.lines_after_imports == -1
    assert var_9.lines_between_sections == 1
    assert var_9.lines_between_types == 0
    assert var_9.combine_as_imports is False
    assert var_9.combine_star is False
    assert var_9.include_trailing_comma is False
    assert var_9.from_first is False
    assert var_9.verbose is False
    assert var_9.quiet is False
    assert var_9.force_adds is False
    assert var_9.force_alphabetical_sort_within_sections is False
    assert var_9.force_alphabetical_sort is False
    assert var_9.force_grid_wrap == 0
    assert var_9.force_sort_within_sections is False
    assert var_9.lexicographical is False
    assert var_9.group_by_package is False
    assert var_9.ignore_whitespace is False
    assert f'{type(var_9.no_lines_before).__module__}.{type(var_9.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.no_lines_before) == 0
    assert var_9.no_inline_sort is False
    assert var_9.ignore_comments is False
    assert var_9.case_sensitive is False
    assert f'{type(var_9.sources).__module__}.{type(var_9.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_9.sources) == 2
    assert var_9.virtual_env == ''
    assert var_9.conda_env == ''
    assert var_9.ensure_newline_before_comments is False
    assert var_9.directory == '/workspace/run'
    assert var_9.profile == ''
    assert var_9.honor_noqa is False
    assert f'{type(var_9.src_paths).__module__}.{type(var_9.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_9.src_paths) == 2
    assert var_9.remove_redundant_aliases is False
    assert var_9.float_to_top is False
    assert var_9.filter_files is False
    assert var_9.formatter == ''
    assert var_9.formatting_function is None
    assert var_9.color_output is False
    assert f'{type(var_9.treat_comments_as_code).__module__}.{type(var_9.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.treat_comments_as_code) == 0
    assert var_9.treat_all_comments_as_code is False
    assert f'{type(var_9.supported_extensions).__module__}.{type(var_9.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.supported_extensions) == 4
    assert f'{type(var_9.blocked_extensions).__module__}.{type(var_9.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.blocked_extensions) == 1
    assert f'{type(var_9.constants).__module__}.{type(var_9.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.constants) == 0
    assert f'{type(var_9.classes).__module__}.{type(var_9.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.classes) == 0
    assert f'{type(var_9.variables).__module__}.{type(var_9.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.variables) == 0
    assert var_9.dedup_headings is False
    assert var_9.only_sections is False
    assert var_9.only_modified is False
    assert var_9.combine_straight_imports is False
    assert var_9.auto_identify_namespace_packages is True
    assert f'{type(var_9.namespace_packages).__module__}.{type(var_9.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_9.namespace_packages) == 0
    assert var_9.follow_links is True
    assert var_9.indented_import_headings is True
    assert var_9.honor_case_in_force_sorted_sections is False
    assert var_9.sort_relative_in_force_sorted_sections is False
    assert var_9.overwrite_in_place is False
    assert var_9.reverse_sort is False
    assert var_9.star_first is False
    assert var_9.git_ls_files == {}
    assert var_9.format_error == '{error}: {message}'
    assert var_9.format_success == '{success}: {message}'
    assert var_9.sort_order == 'natural'
    assert var_9.sort_reexports is False
    assert var_9.split_on_trailing_comma is False
    assert module_2.TYPE_CHECKING is False
    assert module_2.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_2.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.FIRSTPARTY == 'FIRSTPARTY'
    assert module_2.FUTURE == 'FUTURE'
    assert module_2.LOCALFOLDER == 'LOCALFOLDER'
    assert module_2.STDLIB == 'STDLIB'
    assert module_2.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_2.CYTHON_EXTENSIONS).__module__}.{type(module_2.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_2.SUPPORTED_EXTENSIONS).__module__}.{type(module_2.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_2.BLOCKED_EXTENSIONS).__module__}.{type(module_2.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.BLOCKED_EXTENSIONS) == 1
    assert module_2.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_2.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_2.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_2.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_2.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_2.DEFAULT_SKIP).__module__}.{type(module_2.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_SKIP) == 19
    assert module_2.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_2.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_2.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_2.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_2.KNOWN_PREFIX == 'known_'
    assert module_2.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_2.RUNTIME_SOURCE == 'runtime'
    assert module_2.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_2.DEFAULT_CONFIG).__module__}.{type(module_2.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_2.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_2.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_2.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip).__module__}.{type(module_2.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_2.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_2.DEFAULT_CONFIG.skip_gitignore is False
    assert module_2.DEFAULT_CONFIG.line_length == 79
    assert module_2.DEFAULT_CONFIG.wrap_length == 0
    assert module_2.DEFAULT_CONFIG.line_ending == ''
    assert module_2.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_2.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_2.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_2.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_2.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_2.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_2.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_2.DEFAULT_CONFIG.known_other == {}
    assert module_2.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_2.DEFAULT_CONFIG.forced_separate == ()
    assert module_2.DEFAULT_CONFIG.indent == '    '
    assert module_2.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_2.DEFAULT_CONFIG.length_sort is False
    assert module_2.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_2.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_2.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.add_imports).__module__}.{type(module_2.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_2.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.remove_imports) == 0
    assert module_2.DEFAULT_CONFIG.append_only is False
    assert module_2.DEFAULT_CONFIG.reverse_relative is False
    assert module_2.DEFAULT_CONFIG.force_single_line is False
    assert module_2.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_2.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_2.DEFAULT_CONFIG.import_headings == {}
    assert module_2.DEFAULT_CONFIG.import_footers == {}
    assert module_2.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_2.DEFAULT_CONFIG.use_parentheses is False
    assert module_2.DEFAULT_CONFIG.order_by_type is True
    assert module_2.DEFAULT_CONFIG.atomic is False
    assert module_2.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_2.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_2.DEFAULT_CONFIG.lines_between_types == 0
    assert module_2.DEFAULT_CONFIG.combine_as_imports is False
    assert module_2.DEFAULT_CONFIG.combine_star is False
    assert module_2.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_2.DEFAULT_CONFIG.from_first is False
    assert module_2.DEFAULT_CONFIG.verbose is False
    assert module_2.DEFAULT_CONFIG.quiet is False
    assert module_2.DEFAULT_CONFIG.force_adds is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_2.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_2.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_2.DEFAULT_CONFIG.lexicographical is False
    assert module_2.DEFAULT_CONFIG.group_by_package is False
    assert module_2.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_2.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_2.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_2.DEFAULT_CONFIG.no_inline_sort is False
    assert module_2.DEFAULT_CONFIG.ignore_comments is False
    assert module_2.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_2.DEFAULT_CONFIG.sources).__module__}.{type(module_2.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.sources) == 1
    assert module_2.DEFAULT_CONFIG.virtual_env == ''
    assert module_2.DEFAULT_CONFIG.conda_env == ''
    assert module_2.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_2.DEFAULT_CONFIG.directory == '/workspace/run'
    assert module_2.DEFAULT_CONFIG.profile == ''
    assert module_2.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_2.DEFAULT_CONFIG.src_paths).__module__}.{type(module_2.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_2.DEFAULT_CONFIG.src_paths) == 2
    assert module_2.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_2.DEFAULT_CONFIG.float_to_top is False
    assert module_2.DEFAULT_CONFIG.filter_files is False
    assert module_2.DEFAULT_CONFIG.formatter == ''
    assert module_2.DEFAULT_CONFIG.formatting_function is None
    assert module_2.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_2.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_2.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_2.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_2.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_2.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_2.DEFAULT_CONFIG.constants).__module__}.{type(module_2.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.classes).__module__}.{type(module_2.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_2.DEFAULT_CONFIG.variables).__module__}.{type(module_2.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.variables) == 0
    assert module_2.DEFAULT_CONFIG.dedup_headings is False
    assert module_2.DEFAULT_CONFIG.only_sections is False
    assert module_2.DEFAULT_CONFIG.only_modified is False
    assert module_2.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_2.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_2.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_2.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_2.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_2.DEFAULT_CONFIG.follow_links is True
    assert module_2.DEFAULT_CONFIG.indented_import_headings is True
    assert module_2.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_2.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_2.DEFAULT_CONFIG.reverse_sort is False
    assert module_2.DEFAULT_CONFIG.star_first is False
    assert module_2.DEFAULT_CONFIG.git_ls_files == {}
    assert module_2.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_2.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_2.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_2.DEFAULT_CONFIG.sort_reexports is False
    assert module_2.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_2.Config.known_patterns).__module__}.{type(module_2.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments).__module__}.{type(module_2.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.section_comments_end).__module__}.{type(module_2.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skips).__module__}.{type(module_2.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.skip_globs).__module__}.{type(module_2.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_2.Config.sorting_function).__module__}.{type(module_2.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_10 = {}
    var_11 = None
    var_12 = module_0.check_code_string(var_11, file_path=var_11)
    assert var_12 is True
    var_13 = module_0.sort_stream(var_5, var_5, config=var_9, **var_10)
    assert var_13 is False

def test_case_25():
    var_0 = 'import sys\nimport os\n'
    var_1 = [var_0]
    var_2 = {}
    var_3 = True
    var_4 = None
    var_5 = module_0.check_code_string(var_0, var_3, file_path=var_4)
    assert var_5 is False
    assert f'{type(module_0.Empty).__module__}.{type(module_0.Empty).__qualname__}' == 'isort.io._EmptyIO'
    assert f'{type(module_0.CYTHON_EXTENSIONS).__module__}.{type(module_0.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_0.CYTHON_EXTENSIONS) == 2
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
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
    assert module_0.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_6 = module_6.StringIO(*var_1, **var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_7 = []
    var_8 = {}
    var_9 = module_6.StringIO(*var_7, **var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == '_io.StringIO'