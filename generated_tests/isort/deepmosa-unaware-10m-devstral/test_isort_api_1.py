# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.api as module_0
import isort.wrap_modes as module_1
import tokenize as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.check_stream(var_0, extension=var_0)

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

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.check_file(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.sort_file(var_0, show_diff=var_0)

def test_case_4():
    pass

def test_case_5():
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

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '\nCS&O?7U_Bk@mb]_'
    var_1 = None
    module_0.check_stream(var_1, var_1, file_path=var_1, disregard_skip=var_0)

def test_case_7():
    var_0 = '-MqPR&MOAYv!`g`7)H'
    var_1 = None
    var_2 = module_0.check_code_string(var_0, extension=var_1, file_path=var_1, disregard_skip=var_1)
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
    var_3 = module_0.check_code_string(var_0, var_2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = 'p'
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    module_0.check_code_string(var_1, file_path=var_1, **var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = ">\x0bB'$"
    var_2 = '&/(2y\n>gu>\x0b:G8'
    var_3 = {var_2: var_0}
    var_4 = module_0.find_imports_in_paths(var_2, file_path=var_0, unique=var_0)
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
    module_0.sort_code_string(var_1, var_0, var_0, **var_3)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '\x0bykGw^C|\rWc,c^-M<6#'
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
    module_0.sort_stream(var_0, var_0, var_0, var_0, var_0)

def test_case_11():
    var_0 = '\nCS&O?7U_Bk@mb]_'
    var_1 = None
    var_2 = module_2.group()
    assert var_2 == '()'
    assert module_2.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_2.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_2.ENDMARKER == 0
    assert module_2.NAME == 1
    assert module_2.NUMBER == 2
    assert module_2.STRING == 3
    assert module_2.NEWLINE == 4
    assert module_2.INDENT == 5
    assert module_2.DEDENT == 6
    assert module_2.LPAR == 7
    assert module_2.RPAR == 8
    assert module_2.LSQB == 9
    assert module_2.RSQB == 10
    assert module_2.COLON == 11
    assert module_2.COMMA == 12
    assert module_2.SEMI == 13
    assert module_2.PLUS == 14
    assert module_2.MINUS == 15
    assert module_2.STAR == 16
    assert module_2.SLASH == 17
    assert module_2.VBAR == 18
    assert module_2.AMPER == 19
    assert module_2.LESS == 20
    assert module_2.GREATER == 21
    assert module_2.EQUAL == 22
    assert module_2.DOT == 23
    assert module_2.PERCENT == 24
    assert module_2.LBRACE == 25
    assert module_2.RBRACE == 26
    assert module_2.EQEQUAL == 27
    assert module_2.NOTEQUAL == 28
    assert module_2.LESSEQUAL == 29
    assert module_2.GREATEREQUAL == 30
    assert module_2.TILDE == 31
    assert module_2.CIRCUMFLEX == 32
    assert module_2.LEFTSHIFT == 33
    assert module_2.RIGHTSHIFT == 34
    assert module_2.DOUBLESTAR == 35
    assert module_2.PLUSEQUAL == 36
    assert module_2.MINEQUAL == 37
    assert module_2.STAREQUAL == 38
    assert module_2.SLASHEQUAL == 39
    assert module_2.PERCENTEQUAL == 40
    assert module_2.AMPEREQUAL == 41
    assert module_2.VBAREQUAL == 42
    assert module_2.CIRCUMFLEXEQUAL == 43
    assert module_2.LEFTSHIFTEQUAL == 44
    assert module_2.RIGHTSHIFTEQUAL == 45
    assert module_2.DOUBLESTAREQUAL == 46
    assert module_2.DOUBLESLASH == 47
    assert module_2.DOUBLESLASHEQUAL == 48
    assert module_2.AT == 49
    assert module_2.ATEQUAL == 50
    assert module_2.RARROW == 51
    assert module_2.ELLIPSIS == 52
    assert module_2.COLONEQUAL == 53
    assert module_2.OP == 54
    assert module_2.AWAIT == 55
    assert module_2.ASYNC == 56
    assert module_2.TYPE_IGNORE == 57
    assert module_2.TYPE_COMMENT == 58
    assert module_2.SOFT_KEYWORD == 59
    assert module_2.ERRORTOKEN == 60
    assert module_2.COMMENT == 61
    assert module_2.NL == 62
    assert module_2.ENCODING == 63
    assert module_2.N_TOKENS == 64
    assert module_2.NT_OFFSET == 256
    assert module_2.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_2.cookie_re).__module__}.{type(module_2.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.blank_re).__module__}.{type(module_2.blank_re).__qualname__}' == 're.Pattern'
    assert module_2.Whitespace == '[ \\f\\t]*'
    assert module_2.Comment == '#[^\\r\\n]*'
    assert module_2.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_2.Name == '\\w+'
    assert module_2.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_2.Binnumber == '0[bB](?:_?[01])+'
    assert module_2.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_2.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_2.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_2.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_2.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_2.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_2.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_2.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_2.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_2.StringPrefix == '(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)'
    assert module_2.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_2.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_2.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_2.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_2.Triple == '((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'\'\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)""")'
    assert module_2.String == '((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_2.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_2.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_2.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_2.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_2.ContStr == '((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_2.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'\'\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"""))'
    assert module_2.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'\'\'|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|f|b|fR|rF|rb|Br|Rf|F|br|Rb|R|U|fr|FR|rf|RF|BR|bR|u|rB|B|RB|r|Fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_2.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_2.single_quoted == {'fr"', "RB'", "rb'", "fr'", "FR'", 'bR"', 'R"', 'rF"', 'f"', "Fr'", "u'", "bR'", '"', 'b"', 'RB"', 'u"', "Rb'", 'Rf"', 'br"', 'fR"', "RF'", "'", "f'", "U'", "rf'", 'rb"', 'B"', 'Br"', 'U"', 'FR"', 'rf"', "Rf'", 'F"', "Br'", "br'", 'r"', "r'", 'Rb"', "F'", "R'", "rB'", "BR'", 'RF"', "B'", 'Fr"', "fR'", 'BR"', "b'", "rF'", 'rB"'}
    assert module_2.triple_quoted == {"Rb'''", "BR'''", "rb'''", "U'''", 'rB"""', "'''", "F'''", "rf'''", 'br"""', 'u"""', 'fR"""', 'FR"""', "Rf'''", "rB'''", 'RF"""', "bR'''", 'B"""', "fR'''", "RB'''", "b'''", "rF'''", '"""', 'U"""', "R'''", "br'''", "Br'''", 'Fr"""', 'rb"""', 'rF"""', 'R"""', "FR'''", "Fr'''", 'F"""', 'BR"""', 'Rf"""', 'rf"""', 'RB"""', 'Rb"""', 'b"""', 'Br"""', "fr'''", 'bR"""', 'fr"""', "u'''", "f'''", "B'''", 'f"""', "RF'''", "r'''", 'r"""'}
    assert module_2.t == 'Fr'
    assert module_2.u == "Fr'''"
    assert module_2.tabsize == 8
    var_3 = module_0.sort_code_string(var_0, file_path=var_1, show_diff=var_2)
    assert var_3 == ''
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
    var_2 = module_0.sort_code_string(var_0, file_path=var_0, disregard_skip=var_0, show_diff=var_1)
    assert var_2 == ''
    var_3 = 'jVf%g'
    var_4 = {var_3: var_0, var_3: var_0, var_3: var_0}
    module_0.sort_file(var_3, var_0, file_path=var_4, disregard_skip=var_1, show_diff=var_4, **var_4)

def test_case_13():
    var_0 = None
    var_1 = module_0.find_imports_in_code(var_0)
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
    var_2 = var_1.__iter__()
    var_3 = module_0.check_stream(var_2)
    assert var_3 is True
    var_4 = var_2.__ge__(var_2)

def test_case_14():
    var_0 = 'file1.py'
    var_1 = [var_0]
    var_2 = module_0.find_imports_in_paths(var_1)
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
    var_3 = list(var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    module_0.sort_stream(var_0, var_1, file_path=var_1, show_diff=var_1, raise_on_skip=var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = None
    var_2 = module_0.ImportKey.ATTRIBUTE
    var_3 = module_0.find_imports_in_code(var_0, unique=var_2, top_only=var_0)
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
    var_4 = var_3.__iter__()
    var_5 = module_0.check_stream(var_4, var_1)
    assert var_5 is True
    module_0.sort_file(var_4, var_1, var_1, write_to_stdout=var_0, output=var_5)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.find_imports_in_file(var_0)
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
    var_2 = var_1.__iter__()
    var_3 = module_0.find_imports_in_file(var_2, var_2, top_only=var_2)
    var_2.__new__(var_0, *var_2)

def test_case_18():
    var_0 = 'import os\nfrom sys import path\nimport numpy as np\n'
    var_1 = list(var_0)
    var_2 = len(var_1)
    var_3 = 'non_existent_file.py'
    var_4 = module_0.find_imports_in_file(var_3)
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
    var_5 = list(var_4)

def test_case_19():
    var_0 = 'import sys\nfrom collections import defaultdict\n'
    var_1 = [var_0]
    var_2 = True
    var_3 = module_0.find_imports_in_paths(var_1, unique=var_2)
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
    var_4 = list(var_3)
    var_5 = len(var_4)
    var_6 = [var_0]
    var_7 = True
    var_8 = module_0.find_imports_in_paths(var_6, top_only=var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    var_11 = [var_0]
    var_12 = module_0.find_imports_in_paths(var_11, unique=var_7)
    var_13 = list(var_12)
    var_14 = len(var_13)
    var_15 = '/non/existent/path.py'
    var_16 = [var_15]
    var_17 = module_0.find_imports_in_paths(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 0