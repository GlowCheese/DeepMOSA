# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.parse as module_0
import isort.wrap_modes as module_1
import re as module_2
import isort.settings as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.strip_syntax(var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '/\r?%MY'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = None
    var_3 = '$zO#\t>r2F3SS]o'
    var_4 = module_0.strip_syntax(var_3)
    assert var_4 == '$zO# >r2F3SS]o'
    module_0.import_type(var_2)

def test_case_2():
    var_0 = 'from libc.stdlib cimport malloc\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_3():
    var_0 = 'GthH\x0b"o'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'GthH "o'
    assert module_0.TYPE_CHECKING is False
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

def test_case_4():
    var_0 = "Test file_contents with 'as' imports."
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_5():
    var_0 = 'impor os; import ys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_6():
    var_0 = 'Test file_contents preserves comments.'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = "*'"
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = None
    module_0.import_type(var_2)

def test_case_8():
    var_0 = '"""import os"""\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ''
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = '/;j*gEQy2='
    var_3 = module_0.normalize_line(var_2)
    var_4 = None
    var_5 = module_0.file_contents(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    module_2.split(var_4, var_4)

def test_case_10():
    var_0 = 'import os; import sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_11():
    var_0 = 'import os  # isort:skip\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '# =?\x0c=oR"7D<Tjnl'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = '~M3$i_(rsM'
    var_3 = 'dGZ_\rr"Kj}e+\\l7Ym'
    var_4 = None
    var_5 = False
    var_6 = 'Nu(X=`'
    var_7 = (var_6, var_4)
    var_8 = module_0.skip_line(var_3, var_4, var_5, var_7, var_5)
    var_9 = module_0.file_contents(var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = None
    var_11 = module_0.strip_syntax(var_2)
    assert var_11 == '~M3$i_ rsM'
    module_0.import_type(var_10)

def test_case_13():
    var_0 = 'm;\x0ci8'
    var_1 = None
    var_2 = module_0.file_contents(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_2) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_3 = module_0.skip_line(var_0, var_0, var_1, var_1)
    var_4 = False
    var_5 = '\x0cr\t|'
    var_6 = (var_5, var_0)
    var_7 = module_0.skip_line(var_0, var_1, var_4, var_6)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'j#<%\x0b_o lg?&3#N.'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == 'j#<% _o lg?&3#N.'
    assert module_0.TYPE_CHECKING is False
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
    var_2 = module_0.strip_syntax(var_1)
    assert var_2 == 'j#<% _o lg?&3#N.'
    var_3 = None
    var_4 = "|>w''oE/qTpu #"
    var_5 = module_0.skip_line(var_4, var_3, var_3, var_2)
    module_0.file_contents(var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '# =?\x0c=oR"7D<Tjnl'
    var_1 = '`ehvI-\x0bFjk>uMC|;\x0b-'
    var_2 = None
    var_3 = 2334
    var_4 = module_0.skip_line(var_1, var_2, var_3, var_2, var_2)
    assert module_0.TYPE_CHECKING is False
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
    var_5 = module_0.file_contents(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_6 = '~M3$i_(rsM'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = module_0.strip_syntax(var_6)
    assert var_8 == '~M3$i_ rsM'
    module_2.finditer(var_5, var_2)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = var_5.imports
    var_7 = len(var_6)
    var_8 = 'import os  # comment\nfrom sys import argv\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.categorized_comments
    var_11 = len(var_10)
    var_12 = ''
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'x = 1\ny = 2\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = var_15.lines_without_imports
    var_17 = len(var_16)
    var_18 = 'from os import \\\n    path\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = var_19.imports
    var_21 = len(var_20)
    var_22 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = 'import os as operating_system\nfrom sys import argv as args\n'
    var_25 = module_0.file_contents(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_25) == 14
    var_26 = 'straight'
    var_27 = var_25.as_map[var_26]
    var_28 = len(var_27)
    var_29 = 0
    var_30 = 'from'
    var_31 = var_25.as_map[var_30]
    var_32 = len(var_31)
    var_33 = var_32 > var_29
    var_34 = 'import os\n'
    var_35 = module_0.file_contents(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_35) == 14
    var_36 = 'import os; import sys\n'
    var_37 = module_0.file_contents(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_37) == 14
    var_38 = 'import os  # isort:skip\nimport sys\n'
    var_39 = module_0.file_contents(var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_39) == 14
    var_40 = var_39.lines_without_imports
    var_41 = len(var_40)
    var_42 = '# isort:imports-THIRDPARTY\nimport os\n'
    var_43 = module_0.file_contents(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_43) == 14
    var_44 = '"""\nModule docstring\n"""\nimport os\n'
    var_45 = module_0.file_contents(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_45) == 14
    var_46 = 'x = "import os"\nimport sys\n'
    var_47 = module_0.file_contents(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_47) == 14
    var_48 = '\r\n'
    var_49 = module_3.Config()
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'isort.settings.Config'
    assert var_49.py_version == 'py3'
    assert f'{type(var_49.force_to_top).__module__}.{type(var_49.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.force_to_top) == 0
    assert f'{type(var_49.skip).__module__}.{type(var_49.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.skip) == 19
    assert f'{type(var_49.extend_skip).__module__}.{type(var_49.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.extend_skip) == 0
    assert f'{type(var_49.skip_glob).__module__}.{type(var_49.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.skip_glob) == 0
    assert f'{type(var_49.extend_skip_glob).__module__}.{type(var_49.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.extend_skip_glob) == 0
    assert var_49.skip_gitignore is False
    assert var_49.line_length == 79
    assert var_49.wrap_length == 0
    assert var_49.line_ending == ''
    assert var_49.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_49.no_sections is False
    assert f'{type(var_49.known_future_library).__module__}.{type(var_49.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.known_future_library) == 1
    assert f'{type(var_49.known_third_party).__module__}.{type(var_49.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.known_third_party) == 0
    assert f'{type(var_49.known_first_party).__module__}.{type(var_49.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.known_first_party) == 0
    assert f'{type(var_49.known_local_folder).__module__}.{type(var_49.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.known_local_folder) == 0
    assert f'{type(var_49.known_standard_library).__module__}.{type(var_49.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.known_standard_library) == 234
    assert f'{type(var_49.extra_standard_library).__module__}.{type(var_49.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.extra_standard_library) == 0
    assert var_49.known_other == {}
    assert var_49.multi_line_output == module_1.WrapModes.GRID
    assert var_49.forced_separate == ()
    assert var_49.indent == '    '
    assert var_49.comment_prefix == '  #'
    assert var_49.length_sort is False
    assert var_49.length_sort_straight is False
    assert f'{type(var_49.length_sort_sections).__module__}.{type(var_49.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.length_sort_sections) == 0
    assert f'{type(var_49.add_imports).__module__}.{type(var_49.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.add_imports) == 0
    assert f'{type(var_49.remove_imports).__module__}.{type(var_49.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.remove_imports) == 0
    assert var_49.append_only is False
    assert var_49.reverse_relative is False
    assert var_49.force_single_line is False
    assert var_49.single_line_exclusions == ()
    assert var_49.default_section == 'THIRDPARTY'
    assert var_49.import_headings == {}
    assert var_49.import_footers == {}
    assert var_49.balanced_wrapping is False
    assert var_49.use_parentheses is False
    assert var_49.order_by_type is True
    assert var_49.atomic is False
    assert var_49.lines_before_imports == -1
    assert var_49.lines_after_imports == -1
    assert var_49.lines_between_sections == 1
    assert var_49.lines_between_types == 0
    assert var_49.combine_as_imports is False
    assert var_49.combine_star is False
    assert var_49.include_trailing_comma is False
    assert var_49.from_first is False
    assert var_49.verbose is False
    assert var_49.quiet is False
    assert var_49.force_adds is False
    assert var_49.force_alphabetical_sort_within_sections is False
    assert var_49.force_alphabetical_sort is False
    assert var_49.force_grid_wrap == 0
    assert var_49.force_sort_within_sections is False
    assert var_49.lexicographical is False
    assert var_49.group_by_package is False
    assert var_49.ignore_whitespace is False
    assert f'{type(var_49.no_lines_before).__module__}.{type(var_49.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.no_lines_before) == 0
    assert var_49.no_inline_sort is False
    assert var_49.ignore_comments is False
    assert var_49.case_sensitive is False
    assert f'{type(var_49.sources).__module__}.{type(var_49.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_49.sources) == 1
    assert var_49.virtual_env == ''
    assert var_49.conda_env == ''
    assert var_49.ensure_newline_before_comments is False
    assert var_49.directory == '/workspace'
    assert var_49.profile == ''
    assert var_49.honor_noqa is False
    assert f'{type(var_49.src_paths).__module__}.{type(var_49.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_49.src_paths) == 2
    assert var_49.remove_redundant_aliases is False
    assert var_49.float_to_top is False
    assert var_49.filter_files is False
    assert var_49.formatter == ''
    assert var_49.formatting_function is None
    assert var_49.color_output is False
    assert f'{type(var_49.treat_comments_as_code).__module__}.{type(var_49.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.treat_comments_as_code) == 0
    assert var_49.treat_all_comments_as_code is False
    assert f'{type(var_49.supported_extensions).__module__}.{type(var_49.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.supported_extensions) == 4
    assert f'{type(var_49.blocked_extensions).__module__}.{type(var_49.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.blocked_extensions) == 1
    assert f'{type(var_49.constants).__module__}.{type(var_49.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.constants) == 0
    assert f'{type(var_49.classes).__module__}.{type(var_49.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.classes) == 0
    assert f'{type(var_49.variables).__module__}.{type(var_49.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.variables) == 0
    assert var_49.dedup_headings is False
    assert var_49.only_sections is False
    assert var_49.only_modified is False
    assert var_49.combine_straight_imports is False
    assert var_49.auto_identify_namespace_packages is True
    assert f'{type(var_49.namespace_packages).__module__}.{type(var_49.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_49.namespace_packages) == 0
    assert var_49.follow_links is True
    assert var_49.indented_import_headings is True
    assert var_49.honor_case_in_force_sorted_sections is False
    assert var_49.sort_relative_in_force_sorted_sections is False
    assert var_49.overwrite_in_place is False
    assert var_49.reverse_sort is False
    assert var_49.star_first is False
    assert var_49.git_ls_files == {}
    assert var_49.format_error == '{error}: {message}'
    assert var_49.format_success == '{success}: {message}'
    assert var_49.sort_order == 'natural'
    assert var_49.sort_reexports is False
    assert var_49.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_50 = 'import os\r\nimport sys\r\n'
    var_51 = module_0.file_contents(var_50, var_49)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_51) == 14
    var_52 = 'from os import (\n    path,  # path comment\n    getcwd  # getcwd comment\n)\n'
    var_53 = module_0.file_contents(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_53) == 14
    module_2.fullmatch(var_21, var_10)

def test_case_17():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = 'import numpy as np\n'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = var_7.as_map
    var_9 = str(var_8)
    var_10 = 'from os import path, getcwd\n'
    var_11 = module_0.file_contents(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_11) == 14
    var_12 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = var_13.in_lines
    var_15 = len(var_14)
    var_16 = 'from os import \\\n    path, \\\n    getcwd\n'
    var_17 = module_0.file_contents(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_17) == 14
    var_18 = 'def hello():\n    pass\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = 'import os  # operating system\n'
    var_21 = module_0.file_contents(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_21) == 14
    var_22 = 'import os\n\ndef main():\n    pass\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = var_23.lines_without_imports
    var_25 = len(var_24)
    var_26 = ''
    var_27 = module_0.file_contents(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_27) == 14
    var_28 = '\n'
    var_29 = module_0.file_contents(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_29) == 14
    var_30 = 'from os import path,\n'
    var_31 = module_0.file_contents(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_31) == 14
    var_32 = True
    var_33 = module_3.Config()
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'isort.settings.Config'
    assert var_33.py_version == 'py3'
    assert f'{type(var_33.force_to_top).__module__}.{type(var_33.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.force_to_top) == 0
    assert f'{type(var_33.skip).__module__}.{type(var_33.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.skip) == 19
    assert f'{type(var_33.extend_skip).__module__}.{type(var_33.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.extend_skip) == 0
    assert f'{type(var_33.skip_glob).__module__}.{type(var_33.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.skip_glob) == 0
    assert f'{type(var_33.extend_skip_glob).__module__}.{type(var_33.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.extend_skip_glob) == 0
    assert var_33.skip_gitignore is False
    assert var_33.line_length == 79
    assert var_33.wrap_length == 0
    assert var_33.line_ending == ''
    assert var_33.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_33.no_sections is False
    assert f'{type(var_33.known_future_library).__module__}.{type(var_33.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.known_future_library) == 1
    assert f'{type(var_33.known_third_party).__module__}.{type(var_33.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.known_third_party) == 0
    assert f'{type(var_33.known_first_party).__module__}.{type(var_33.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.known_first_party) == 0
    assert f'{type(var_33.known_local_folder).__module__}.{type(var_33.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.known_local_folder) == 0
    assert f'{type(var_33.known_standard_library).__module__}.{type(var_33.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.known_standard_library) == 234
    assert f'{type(var_33.extra_standard_library).__module__}.{type(var_33.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.extra_standard_library) == 0
    assert var_33.known_other == {}
    assert var_33.multi_line_output == module_1.WrapModes.GRID
    assert var_33.forced_separate == ()
    assert var_33.indent == '    '
    assert var_33.comment_prefix == '  #'
    assert var_33.length_sort is False
    assert var_33.length_sort_straight is False
    assert f'{type(var_33.length_sort_sections).__module__}.{type(var_33.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.length_sort_sections) == 0
    assert f'{type(var_33.add_imports).__module__}.{type(var_33.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.add_imports) == 0
    assert f'{type(var_33.remove_imports).__module__}.{type(var_33.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.remove_imports) == 0
    assert var_33.append_only is False
    assert var_33.reverse_relative is False
    assert var_33.force_single_line is False
    assert var_33.single_line_exclusions == ()
    assert var_33.default_section == 'THIRDPARTY'
    assert var_33.import_headings == {}
    assert var_33.import_footers == {}
    assert var_33.balanced_wrapping is False
    assert var_33.use_parentheses is False
    assert var_33.order_by_type is True
    assert var_33.atomic is False
    assert var_33.lines_before_imports == -1
    assert var_33.lines_after_imports == -1
    assert var_33.lines_between_sections == 1
    assert var_33.lines_between_types == 0
    assert var_33.combine_as_imports is False
    assert var_33.combine_star is False
    assert var_33.include_trailing_comma is False
    assert var_33.from_first is False
    assert var_33.verbose is False
    assert var_33.quiet is False
    assert var_33.force_adds is False
    assert var_33.force_alphabetical_sort_within_sections is False
    assert var_33.force_alphabetical_sort is False
    assert var_33.force_grid_wrap == 0
    assert var_33.force_sort_within_sections is False
    assert var_33.lexicographical is False
    assert var_33.group_by_package is False
    assert var_33.ignore_whitespace is False
    assert f'{type(var_33.no_lines_before).__module__}.{type(var_33.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.no_lines_before) == 0
    assert var_33.no_inline_sort is False
    assert var_33.ignore_comments is False
    assert var_33.case_sensitive is False
    assert f'{type(var_33.sources).__module__}.{type(var_33.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_33.sources) == 1
    assert var_33.virtual_env == ''
    assert var_33.conda_env == ''
    assert var_33.ensure_newline_before_comments is False
    assert var_33.directory == '/workspace'
    assert var_33.profile == ''
    assert var_33.honor_noqa is False
    assert f'{type(var_33.src_paths).__module__}.{type(var_33.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_33.src_paths) == 2
    assert var_33.remove_redundant_aliases is False
    assert var_33.float_to_top is False
    assert var_33.filter_files is False
    assert var_33.formatter == ''
    assert var_33.formatting_function is None
    assert var_33.color_output is False
    assert f'{type(var_33.treat_comments_as_code).__module__}.{type(var_33.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.treat_comments_as_code) == 0
    assert var_33.treat_all_comments_as_code is False
    assert f'{type(var_33.supported_extensions).__module__}.{type(var_33.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.supported_extensions) == 4
    assert f'{type(var_33.blocked_extensions).__module__}.{type(var_33.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.blocked_extensions) == 1
    assert f'{type(var_33.constants).__module__}.{type(var_33.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.constants) == 0
    assert f'{type(var_33.classes).__module__}.{type(var_33.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.classes) == 0
    assert f'{type(var_33.variables).__module__}.{type(var_33.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.variables) == 0
    assert var_33.dedup_headings is False
    assert var_33.only_sections is False
    assert var_33.only_modified is False
    assert var_33.combine_straight_imports is False
    assert var_33.auto_identify_namespace_packages is True
    assert f'{type(var_33.namespace_packages).__module__}.{type(var_33.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_33.namespace_packages) == 0
    assert var_33.follow_links is True
    assert var_33.indented_import_headings is True
    assert var_33.honor_case_in_force_sorted_sections is False
    assert var_33.sort_relative_in_force_sorted_sections is False
    assert var_33.overwrite_in_place is False
    assert var_33.reverse_sort is False
    assert var_33.star_first is False
    assert var_33.git_ls_files == {}
    assert var_33.format_error == '{error}: {message}'
    assert var_33.format_success == '{success}: {message}'
    assert var_33.sort_order == 'natural'
    assert var_33.sort_reexports is False
    assert var_33.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_34 = 'from os import path, getcwd\n'
    var_35 = module_0.file_contents(var_34, var_33)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_35) == 14
    var_36 = module_3.Config()
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.settings.Config'
    assert var_36.py_version == 'py3'
    assert f'{type(var_36.force_to_top).__module__}.{type(var_36.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.force_to_top) == 0
    assert f'{type(var_36.skip).__module__}.{type(var_36.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.skip) == 19
    assert f'{type(var_36.extend_skip).__module__}.{type(var_36.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.extend_skip) == 0
    assert f'{type(var_36.skip_glob).__module__}.{type(var_36.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.skip_glob) == 0
    assert f'{type(var_36.extend_skip_glob).__module__}.{type(var_36.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.extend_skip_glob) == 0
    assert var_36.skip_gitignore is False
    assert var_36.line_length == 79
    assert var_36.wrap_length == 0
    assert var_36.line_ending == ''
    assert var_36.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_36.no_sections is False
    assert f'{type(var_36.known_future_library).__module__}.{type(var_36.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.known_future_library) == 1
    assert f'{type(var_36.known_third_party).__module__}.{type(var_36.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.known_third_party) == 0
    assert f'{type(var_36.known_first_party).__module__}.{type(var_36.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.known_first_party) == 0
    assert f'{type(var_36.known_local_folder).__module__}.{type(var_36.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.known_local_folder) == 0
    assert f'{type(var_36.known_standard_library).__module__}.{type(var_36.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.known_standard_library) == 234
    assert f'{type(var_36.extra_standard_library).__module__}.{type(var_36.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.extra_standard_library) == 0
    assert var_36.known_other == {}
    assert var_36.multi_line_output == module_1.WrapModes.GRID
    assert var_36.forced_separate == ()
    assert var_36.indent == '    '
    assert var_36.comment_prefix == '  #'
    assert var_36.length_sort is False
    assert var_36.length_sort_straight is False
    assert f'{type(var_36.length_sort_sections).__module__}.{type(var_36.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.length_sort_sections) == 0
    assert f'{type(var_36.add_imports).__module__}.{type(var_36.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.add_imports) == 0
    assert f'{type(var_36.remove_imports).__module__}.{type(var_36.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.remove_imports) == 0
    assert var_36.append_only is False
    assert var_36.reverse_relative is False
    assert var_36.force_single_line is False
    assert var_36.single_line_exclusions == ()
    assert var_36.default_section == 'THIRDPARTY'
    assert var_36.import_headings == {}
    assert var_36.import_footers == {}
    assert var_36.balanced_wrapping is False
    assert var_36.use_parentheses is False
    assert var_36.order_by_type is True
    assert var_36.atomic is False
    assert var_36.lines_before_imports == -1
    assert var_36.lines_after_imports == -1
    assert var_36.lines_between_sections == 1
    assert var_36.lines_between_types == 0
    assert var_36.combine_as_imports is False
    assert var_36.combine_star is False
    assert var_36.include_trailing_comma is False
    assert var_36.from_first is False
    assert var_36.verbose is False
    assert var_36.quiet is False
    assert var_36.force_adds is False
    assert var_36.force_alphabetical_sort_within_sections is False
    assert var_36.force_alphabetical_sort is False
    assert var_36.force_grid_wrap == 0
    assert var_36.force_sort_within_sections is False
    assert var_36.lexicographical is False
    assert var_36.group_by_package is False
    assert var_36.ignore_whitespace is False
    assert f'{type(var_36.no_lines_before).__module__}.{type(var_36.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.no_lines_before) == 0
    assert var_36.no_inline_sort is False
    assert var_36.ignore_comments is False
    assert var_36.case_sensitive is False
    assert f'{type(var_36.sources).__module__}.{type(var_36.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_36.sources) == 1
    assert var_36.virtual_env == ''
    assert var_36.conda_env == ''
    assert var_36.ensure_newline_before_comments is False
    assert var_36.directory == '/workspace'
    assert var_36.profile == ''
    assert var_36.honor_noqa is False
    assert f'{type(var_36.src_paths).__module__}.{type(var_36.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_36.src_paths) == 2
    assert var_36.remove_redundant_aliases is False
    assert var_36.float_to_top is False
    assert var_36.filter_files is False
    assert var_36.formatter == ''
    assert var_36.formatting_function is None
    assert var_36.color_output is False
    assert f'{type(var_36.treat_comments_as_code).__module__}.{type(var_36.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.treat_comments_as_code) == 0
    assert var_36.treat_all_comments_as_code is False
    assert f'{type(var_36.supported_extensions).__module__}.{type(var_36.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.supported_extensions) == 4
    assert f'{type(var_36.blocked_extensions).__module__}.{type(var_36.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.blocked_extensions) == 1
    assert f'{type(var_36.constants).__module__}.{type(var_36.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.constants) == 0
    assert f'{type(var_36.classes).__module__}.{type(var_36.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.classes) == 0
    assert f'{type(var_36.variables).__module__}.{type(var_36.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.variables) == 0
    assert var_36.dedup_headings is False
    assert var_36.only_sections is False
    assert var_36.only_modified is False
    assert var_36.combine_straight_imports is False
    assert var_36.auto_identify_namespace_packages is True
    assert f'{type(var_36.namespace_packages).__module__}.{type(var_36.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_36.namespace_packages) == 0
    assert var_36.follow_links is True
    assert var_36.indented_import_headings is True
    assert var_36.honor_case_in_force_sorted_sections is False
    assert var_36.sort_relative_in_force_sorted_sections is False
    assert var_36.overwrite_in_place is False
    assert var_36.reverse_sort is False
    assert var_36.star_first is False
    assert var_36.git_ls_files == {}
    assert var_36.format_error == '{error}: {message}'
    assert var_36.format_success == '{success}: {message}'
    assert var_36.sort_order == 'natural'
    assert var_36.sort_reexports is False
    assert var_36.split_on_trailing_comma is False
    var_37 = '# isort: split\nimport os\n'
    var_38 = module_0.file_contents(var_37, var_36)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_38) == 14
    var_39 = var_38.in_lines
    var_40 = len(var_39)
    var_41 = 'import os; import sys\n'
    var_42 = module_0.file_contents(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_42) == 14
    var_43 = 'from os import (\n    path,  # path module\n    getcwd\n)\n'
    var_44 = module_0.file_contents(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_44) == 14
    var_45 = var_44.categorized_comments
    var_46 = 'import os\n'
    var_47 = module_0.file_contents(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_47) == 14
    var_48 = 'in_lines'
    var_49 = hasattr(var_47, var_48)
    var_50 = 'lines_without_imports'
    var_51 = hasattr(var_47, var_50)
    var_52 = hasattr(var_47, var_16)
    var_53 = 'imports'
    var_54 = hasattr(var_47, var_53)
    var_55 = 'as_map'
    var_56 = hasattr(var_47, var_55)
    var_57 = 'categorized_comments'
    var_58 = hasattr(var_47, var_57)
    var_59 = 'trailing_commas'
    var_60 = hasattr(var_47, var_59)
    var_61 = 'verbose_output'
    var_62 = hasattr(var_47, var_61)
    var_63 = module_3.Config()
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'isort.settings.Config'
    assert var_63.py_version == 'py3'
    assert f'{type(var_63.force_to_top).__module__}.{type(var_63.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.force_to_top) == 0
    assert f'{type(var_63.skip).__module__}.{type(var_63.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.skip) == 19
    assert f'{type(var_63.extend_skip).__module__}.{type(var_63.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.extend_skip) == 0
    assert f'{type(var_63.skip_glob).__module__}.{type(var_63.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.skip_glob) == 0
    assert f'{type(var_63.extend_skip_glob).__module__}.{type(var_63.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.extend_skip_glob) == 0
    assert var_63.skip_gitignore is False
    assert var_63.line_length == 79
    assert var_63.wrap_length == 0
    assert var_63.line_ending == ''
    assert var_63.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_63.no_sections is False
    assert f'{type(var_63.known_future_library).__module__}.{type(var_63.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.known_future_library) == 1
    assert f'{type(var_63.known_third_party).__module__}.{type(var_63.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.known_third_party) == 0
    assert f'{type(var_63.known_first_party).__module__}.{type(var_63.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.known_first_party) == 0
    assert f'{type(var_63.known_local_folder).__module__}.{type(var_63.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.known_local_folder) == 0
    assert f'{type(var_63.known_standard_library).__module__}.{type(var_63.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.known_standard_library) == 234
    assert f'{type(var_63.extra_standard_library).__module__}.{type(var_63.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.extra_standard_library) == 0
    assert var_63.known_other == {}
    assert var_63.multi_line_output == module_1.WrapModes.GRID
    assert var_63.forced_separate == ()
    assert var_63.indent == '    '
    assert var_63.comment_prefix == '  #'
    assert var_63.length_sort is False
    assert var_63.length_sort_straight is False
    assert f'{type(var_63.length_sort_sections).__module__}.{type(var_63.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.length_sort_sections) == 0
    assert f'{type(var_63.add_imports).__module__}.{type(var_63.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.add_imports) == 0
    assert f'{type(var_63.remove_imports).__module__}.{type(var_63.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.remove_imports) == 0
    assert var_63.append_only is False
    assert var_63.reverse_relative is False
    assert var_63.force_single_line is False
    assert var_63.single_line_exclusions == ()
    assert var_63.default_section == 'THIRDPARTY'
    assert var_63.import_headings == {}
    assert var_63.import_footers == {}
    assert var_63.balanced_wrapping is False
    assert var_63.use_parentheses is False
    assert var_63.order_by_type is True
    assert var_63.atomic is False
    assert var_63.lines_before_imports == -1
    assert var_63.lines_after_imports == -1
    assert var_63.lines_between_sections == 1
    assert var_63.lines_between_types == 0
    assert var_63.combine_as_imports is False
    assert var_63.combine_star is False
    assert var_63.include_trailing_comma is False
    assert var_63.from_first is False
    assert var_63.verbose is False
    assert var_63.quiet is False
    assert var_63.force_adds is False
    assert var_63.force_alphabetical_sort_within_sections is False
    assert var_63.force_alphabetical_sort is False
    assert var_63.force_grid_wrap == 0
    assert var_63.force_sort_within_sections is False
    assert var_63.lexicographical is False
    assert var_63.group_by_package is False
    assert var_63.ignore_whitespace is False
    assert f'{type(var_63.no_lines_before).__module__}.{type(var_63.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.no_lines_before) == 0
    assert var_63.no_inline_sort is False
    assert var_63.ignore_comments is False
    assert var_63.case_sensitive is False
    assert f'{type(var_63.sources).__module__}.{type(var_63.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_63.sources) == 1
    assert var_63.virtual_env == ''
    assert var_63.conda_env == ''
    assert var_63.ensure_newline_before_comments is False
    assert var_63.directory == '/workspace'
    assert var_63.profile == ''
    assert var_63.honor_noqa is False
    assert f'{type(var_63.src_paths).__module__}.{type(var_63.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_63.src_paths) == 2
    assert var_63.remove_redundant_aliases is False
    assert var_63.float_to_top is False
    assert var_63.filter_files is False
    assert var_63.formatter == ''
    assert var_63.formatting_function is None
    assert var_63.color_output is False
    assert f'{type(var_63.treat_comments_as_code).__module__}.{type(var_63.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.treat_comments_as_code) == 0
    assert var_63.treat_all_comments_as_code is False
    assert f'{type(var_63.supported_extensions).__module__}.{type(var_63.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.supported_extensions) == 4
    assert f'{type(var_63.blocked_extensions).__module__}.{type(var_63.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.blocked_extensions) == 1
    assert f'{type(var_63.constants).__module__}.{type(var_63.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.constants) == 0
    assert f'{type(var_63.classes).__module__}.{type(var_63.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.classes) == 0
    assert f'{type(var_63.variables).__module__}.{type(var_63.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.variables) == 0
    assert var_63.dedup_headings is False
    assert var_63.only_sections is False
    assert var_63.only_modified is False
    assert var_63.combine_straight_imports is False
    assert var_63.auto_identify_namespace_packages is True
    assert f'{type(var_63.namespace_packages).__module__}.{type(var_63.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_63.namespace_packages) == 0
    assert var_63.follow_links is True
    assert var_63.indented_import_headings is True
    assert var_63.honor_case_in_force_sorted_sections is False
    assert var_63.sort_relative_in_force_sorted_sections is False
    assert var_63.overwrite_in_place is False
    assert var_63.reverse_sort is False
    assert var_63.star_first is False
    assert var_63.git_ls_files == {}
    assert var_63.format_error == '{error}: {message}'
    assert var_63.format_success == '{success}: {message}'
    assert var_63.sort_order == 'natural'
    assert var_63.sort_reexports is False
    assert var_63.split_on_trailing_comma is False
    var_64 = 'import os as os\n'
    var_65 = module_0.file_contents(var_64, var_63)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_65) == 14

def test_case_18():
    var_0 = 'import numpy as np\nfrom os import path as p\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_19():
    var_0 = 'import os, \\\n    sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

def test_case_20():
    var_0 = 'from os import (\n    path,\n    environ\n)\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports

def test_case_21():
    var_0 = '# isort: third_party'
    var_1 = [var_0]
    var_2 = module_3.Config()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.settings.Config'
    assert var_2.py_version == 'py3'
    assert f'{type(var_2.force_to_top).__module__}.{type(var_2.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.force_to_top) == 0
    assert f'{type(var_2.skip).__module__}.{type(var_2.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip) == 19
    assert f'{type(var_2.extend_skip).__module__}.{type(var_2.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip) == 0
    assert f'{type(var_2.skip_glob).__module__}.{type(var_2.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.skip_glob) == 0
    assert f'{type(var_2.extend_skip_glob).__module__}.{type(var_2.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extend_skip_glob) == 0
    assert var_2.skip_gitignore is False
    assert var_2.line_length == 79
    assert var_2.wrap_length == 0
    assert var_2.line_ending == ''
    assert var_2.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_2.no_sections is False
    assert f'{type(var_2.known_future_library).__module__}.{type(var_2.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_future_library) == 1
    assert f'{type(var_2.known_third_party).__module__}.{type(var_2.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_third_party) == 0
    assert f'{type(var_2.known_first_party).__module__}.{type(var_2.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_first_party) == 0
    assert f'{type(var_2.known_local_folder).__module__}.{type(var_2.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_local_folder) == 0
    assert f'{type(var_2.known_standard_library).__module__}.{type(var_2.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.known_standard_library) == 234
    assert f'{type(var_2.extra_standard_library).__module__}.{type(var_2.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.extra_standard_library) == 0
    assert var_2.known_other == {}
    assert var_2.multi_line_output == module_1.WrapModes.GRID
    assert var_2.forced_separate == ()
    assert var_2.indent == '    '
    assert var_2.comment_prefix == '  #'
    assert var_2.length_sort is False
    assert var_2.length_sort_straight is False
    assert f'{type(var_2.length_sort_sections).__module__}.{type(var_2.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.length_sort_sections) == 0
    assert f'{type(var_2.add_imports).__module__}.{type(var_2.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.add_imports) == 0
    assert f'{type(var_2.remove_imports).__module__}.{type(var_2.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.remove_imports) == 0
    assert var_2.append_only is False
    assert var_2.reverse_relative is False
    assert var_2.force_single_line is False
    assert var_2.single_line_exclusions == ()
    assert var_2.default_section == 'THIRDPARTY'
    assert var_2.import_headings == {}
    assert var_2.import_footers == {}
    assert var_2.balanced_wrapping is False
    assert var_2.use_parentheses is False
    assert var_2.order_by_type is True
    assert var_2.atomic is False
    assert var_2.lines_before_imports == -1
    assert var_2.lines_after_imports == -1
    assert var_2.lines_between_sections == 1
    assert var_2.lines_between_types == 0
    assert var_2.combine_as_imports is False
    assert var_2.combine_star is False
    assert var_2.include_trailing_comma is False
    assert var_2.from_first is False
    assert var_2.verbose is False
    assert var_2.quiet is False
    assert var_2.force_adds is False
    assert var_2.force_alphabetical_sort_within_sections is False
    assert var_2.force_alphabetical_sort is False
    assert var_2.force_grid_wrap == 0
    assert var_2.force_sort_within_sections is False
    assert var_2.lexicographical is False
    assert var_2.group_by_package is False
    assert var_2.ignore_whitespace is False
    assert f'{type(var_2.no_lines_before).__module__}.{type(var_2.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.no_lines_before) == 0
    assert var_2.no_inline_sort is False
    assert var_2.ignore_comments is False
    assert var_2.case_sensitive is False
    assert f'{type(var_2.sources).__module__}.{type(var_2.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_2.sources) == 1
    assert var_2.virtual_env == ''
    assert var_2.conda_env == ''
    assert var_2.ensure_newline_before_comments is False
    assert var_2.directory == '/workspace'
    assert var_2.profile == ''
    assert var_2.honor_noqa is False
    assert f'{type(var_2.src_paths).__module__}.{type(var_2.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_2.src_paths) == 2
    assert var_2.remove_redundant_aliases is False
    assert var_2.float_to_top is False
    assert var_2.filter_files is False
    assert var_2.formatter == ''
    assert var_2.formatting_function is None
    assert var_2.color_output is False
    assert f'{type(var_2.treat_comments_as_code).__module__}.{type(var_2.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.treat_comments_as_code) == 0
    assert var_2.treat_all_comments_as_code is False
    assert f'{type(var_2.supported_extensions).__module__}.{type(var_2.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.supported_extensions) == 4
    assert f'{type(var_2.blocked_extensions).__module__}.{type(var_2.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.blocked_extensions) == 1
    assert f'{type(var_2.constants).__module__}.{type(var_2.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.constants) == 0
    assert f'{type(var_2.classes).__module__}.{type(var_2.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.classes) == 0
    assert f'{type(var_2.variables).__module__}.{type(var_2.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.variables) == 0
    assert var_2.dedup_headings is False
    assert var_2.only_sections is False
    assert var_2.only_modified is False
    assert var_2.combine_straight_imports is False
    assert var_2.auto_identify_namespace_packages is True
    assert f'{type(var_2.namespace_packages).__module__}.{type(var_2.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_2.namespace_packages) == 0
    assert var_2.follow_links is True
    assert var_2.indented_import_headings is True
    assert var_2.honor_case_in_force_sorted_sections is False
    assert var_2.sort_relative_in_force_sorted_sections is False
    assert var_2.overwrite_in_place is False
    assert var_2.reverse_sort is False
    assert var_2.star_first is False
    assert var_2.git_ls_files == {}
    assert var_2.format_error == '{error}: {message}'
    assert var_2.format_success == '{success}: {message}'
    assert var_2.sort_order == 'natural'
    assert var_2.sort_reexports is False
    assert var_2.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_3 = '# isort: third_party\nimport numpy\n'
    var_4 = module_0.file_contents(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_4) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'Test the file_contents function with various import scenarios.'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = var_5.imports
    var_7 = len(var_6)
    var_8 = module_0.file_contents(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_8) == 14
    var_9 = var_8.categorized_comments
    var_10 = len(var_9)
    var_11 = ''
    var_12 = module_0.file_contents(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_12) == 14
    var_13 = '='
    var_14 = module_0.file_contents(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_14) == 14
    var_15 = var_14.lines_without_imports
    var_16 = len(var_15)
    var_17 = 'from os import \\\n    path\n'
    var_18 = module_0.file_contents(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_18) == 14
    var_19 = var_18.imports
    var_20 = len(var_19)
    var_21 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_22 = module_0.file_contents(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_22) == 14
    var_23 = 'import os as operating_system\nfrom sys import argv as args\n'
    var_24 = module_0.file_contents(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_24) == 14
    var_25 = 'straight'
    var_26 = var_24.as_map[var_25]
    var_27 = len(var_26)
    var_28 = 0
    var_29 = var_27 > var_28
    var_30 = 'from'
    var_31 = var_24.as_map[var_30]
    var_32 = len(var_31)
    var_33 = 'import os\n'
    var_34 = module_0.file_contents(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_34) == 14
    var_35 = 'import os; import sys\n'
    var_36 = module_0.file_contents(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = 'import os  # isort:skip\nimport sys\n'
    var_38 = module_0.file_contents(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_38) == 14
    var_39 = var_38.lines_without_imports
    var_40 = len(var_39)
    var_41 = '"""\nModule docstring\n"""\nimport os\n'
    var_42 = module_0.file_contents(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_42) == 14
    var_43 = '\r\n'
    var_44 = module_3.Config()
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'isort.settings.Config'
    assert var_44.py_version == 'py3'
    assert f'{type(var_44.force_to_top).__module__}.{type(var_44.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.force_to_top) == 0
    assert f'{type(var_44.skip).__module__}.{type(var_44.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.skip) == 19
    assert f'{type(var_44.extend_skip).__module__}.{type(var_44.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.extend_skip) == 0
    assert f'{type(var_44.skip_glob).__module__}.{type(var_44.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.skip_glob) == 0
    assert f'{type(var_44.extend_skip_glob).__module__}.{type(var_44.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.extend_skip_glob) == 0
    assert var_44.skip_gitignore is False
    assert var_44.line_length == 79
    assert var_44.wrap_length == 0
    assert var_44.line_ending == ''
    assert var_44.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_44.no_sections is False
    assert f'{type(var_44.known_future_library).__module__}.{type(var_44.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.known_future_library) == 1
    assert f'{type(var_44.known_third_party).__module__}.{type(var_44.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.known_third_party) == 0
    assert f'{type(var_44.known_first_party).__module__}.{type(var_44.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.known_first_party) == 0
    assert f'{type(var_44.known_local_folder).__module__}.{type(var_44.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.known_local_folder) == 0
    assert f'{type(var_44.known_standard_library).__module__}.{type(var_44.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.known_standard_library) == 234
    assert f'{type(var_44.extra_standard_library).__module__}.{type(var_44.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.extra_standard_library) == 0
    assert var_44.known_other == {}
    assert var_44.multi_line_output == module_1.WrapModes.GRID
    assert var_44.forced_separate == ()
    assert var_44.indent == '    '
    assert var_44.comment_prefix == '  #'
    assert var_44.length_sort is False
    assert var_44.length_sort_straight is False
    assert f'{type(var_44.length_sort_sections).__module__}.{type(var_44.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.length_sort_sections) == 0
    assert f'{type(var_44.add_imports).__module__}.{type(var_44.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.add_imports) == 0
    assert f'{type(var_44.remove_imports).__module__}.{type(var_44.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.remove_imports) == 0
    assert var_44.append_only is False
    assert var_44.reverse_relative is False
    assert var_44.force_single_line is False
    assert var_44.single_line_exclusions == ()
    assert var_44.default_section == 'THIRDPARTY'
    assert var_44.import_headings == {}
    assert var_44.import_footers == {}
    assert var_44.balanced_wrapping is False
    assert var_44.use_parentheses is False
    assert var_44.order_by_type is True
    assert var_44.atomic is False
    assert var_44.lines_before_imports == -1
    assert var_44.lines_after_imports == -1
    assert var_44.lines_between_sections == 1
    assert var_44.lines_between_types == 0
    assert var_44.combine_as_imports is False
    assert var_44.combine_star is False
    assert var_44.include_trailing_comma is False
    assert var_44.from_first is False
    assert var_44.verbose is False
    assert var_44.quiet is False
    assert var_44.force_adds is False
    assert var_44.force_alphabetical_sort_within_sections is False
    assert var_44.force_alphabetical_sort is False
    assert var_44.force_grid_wrap == 0
    assert var_44.force_sort_within_sections is False
    assert var_44.lexicographical is False
    assert var_44.group_by_package is False
    assert var_44.ignore_whitespace is False
    assert f'{type(var_44.no_lines_before).__module__}.{type(var_44.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.no_lines_before) == 0
    assert var_44.no_inline_sort is False
    assert var_44.ignore_comments is False
    assert var_44.case_sensitive is False
    assert f'{type(var_44.sources).__module__}.{type(var_44.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_44.sources) == 1
    assert var_44.virtual_env == ''
    assert var_44.conda_env == ''
    assert var_44.ensure_newline_before_comments is False
    assert var_44.directory == '/workspace'
    assert var_44.profile == ''
    assert var_44.honor_noqa is False
    assert f'{type(var_44.src_paths).__module__}.{type(var_44.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_44.src_paths) == 2
    assert var_44.remove_redundant_aliases is False
    assert var_44.float_to_top is False
    assert var_44.filter_files is False
    assert var_44.formatter == ''
    assert var_44.formatting_function is None
    assert var_44.color_output is False
    assert f'{type(var_44.treat_comments_as_code).__module__}.{type(var_44.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.treat_comments_as_code) == 0
    assert var_44.treat_all_comments_as_code is False
    assert f'{type(var_44.supported_extensions).__module__}.{type(var_44.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.supported_extensions) == 4
    assert f'{type(var_44.blocked_extensions).__module__}.{type(var_44.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.blocked_extensions) == 1
    assert f'{type(var_44.constants).__module__}.{type(var_44.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.constants) == 0
    assert f'{type(var_44.classes).__module__}.{type(var_44.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.classes) == 0
    assert f'{type(var_44.variables).__module__}.{type(var_44.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.variables) == 0
    assert var_44.dedup_headings is False
    assert var_44.only_sections is False
    assert var_44.only_modified is False
    assert var_44.combine_straight_imports is False
    assert var_44.auto_identify_namespace_packages is True
    assert f'{type(var_44.namespace_packages).__module__}.{type(var_44.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_44.namespace_packages) == 0
    assert var_44.follow_links is True
    assert var_44.indented_import_headings is True
    assert var_44.honor_case_in_force_sorted_sections is False
    assert var_44.sort_relative_in_force_sorted_sections is False
    assert var_44.overwrite_in_place is False
    assert var_44.reverse_sort is False
    assert var_44.star_first is False
    assert var_44.git_ls_files == {}
    assert var_44.format_error == '{error}: {message}'
    assert var_44.format_success == '{success}: {message}'
    assert var_44.sort_order == 'natural'
    assert var_44.sort_reexports is False
    assert var_44.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_45 = 'import os\r\nimport sys\r\n'
    var_46 = module_0.file_contents(var_45, var_44)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_46) == 14
    var_47 = 'from os import (\n    path,  # path comment\n    getcwd  # getcwd comment\n)\n'
    var_48 = module_0.file_contents(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_48) == 14
    var_49 = 'from os import (\n    path,\n)\n'
    var_50 = module_0.file_contents(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_50) == 14
    var_51 = var_50.trailing_commas
    var_52 = len(var_51)
    var_53 = var_52 >= var_28
    var_54 = 'from os import path\nfrom os import getcwd\n'
    var_55 = module_0.file_contents(var_54)
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_55) == 14
    var_56 = 'import os; import sys  # comment\n'
    var_57 = module_0.file_contents(var_56)
    assert f'{type(var_57).__module__}.{type(var_57).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_57) == 14
    var_58 = True
    var_59 = module_3.Config()
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'isort.settings.Config'
    assert var_59.py_version == 'py3'
    assert f'{type(var_59.force_to_top).__module__}.{type(var_59.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.force_to_top) == 0
    assert f'{type(var_59.skip).__module__}.{type(var_59.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.skip) == 19
    assert f'{type(var_59.extend_skip).__module__}.{type(var_59.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.extend_skip) == 0
    assert f'{type(var_59.skip_glob).__module__}.{type(var_59.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.skip_glob) == 0
    assert f'{type(var_59.extend_skip_glob).__module__}.{type(var_59.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.extend_skip_glob) == 0
    assert var_59.skip_gitignore is False
    assert var_59.line_length == 79
    assert var_59.wrap_length == 0
    assert var_59.line_ending == ''
    assert var_59.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_59.no_sections is False
    assert f'{type(var_59.known_future_library).__module__}.{type(var_59.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.known_future_library) == 1
    assert f'{type(var_59.known_third_party).__module__}.{type(var_59.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.known_third_party) == 0
    assert f'{type(var_59.known_first_party).__module__}.{type(var_59.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.known_first_party) == 0
    assert f'{type(var_59.known_local_folder).__module__}.{type(var_59.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.known_local_folder) == 0
    assert f'{type(var_59.known_standard_library).__module__}.{type(var_59.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.known_standard_library) == 234
    assert f'{type(var_59.extra_standard_library).__module__}.{type(var_59.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.extra_standard_library) == 0
    assert var_59.known_other == {}
    assert var_59.multi_line_output == module_1.WrapModes.GRID
    assert var_59.forced_separate == ()
    assert var_59.indent == '    '
    assert var_59.comment_prefix == '  #'
    assert var_59.length_sort is False
    assert var_59.length_sort_straight is False
    assert f'{type(var_59.length_sort_sections).__module__}.{type(var_59.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.length_sort_sections) == 0
    assert f'{type(var_59.add_imports).__module__}.{type(var_59.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.add_imports) == 0
    assert f'{type(var_59.remove_imports).__module__}.{type(var_59.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.remove_imports) == 0
    assert var_59.append_only is False
    assert var_59.reverse_relative is False
    assert var_59.force_single_line is False
    assert var_59.single_line_exclusions == ()
    assert var_59.default_section == 'THIRDPARTY'
    assert var_59.import_headings == {}
    assert var_59.import_footers == {}
    assert var_59.balanced_wrapping is False
    assert var_59.use_parentheses is False
    assert var_59.order_by_type is True
    assert var_59.atomic is False
    assert var_59.lines_before_imports == -1
    assert var_59.lines_after_imports == -1
    assert var_59.lines_between_sections == 1
    assert var_59.lines_between_types == 0
    assert var_59.combine_as_imports is False
    assert var_59.combine_star is False
    assert var_59.include_trailing_comma is False
    assert var_59.from_first is False
    assert var_59.verbose is False
    assert var_59.quiet is False
    assert var_59.force_adds is False
    assert var_59.force_alphabetical_sort_within_sections is False
    assert var_59.force_alphabetical_sort is False
    assert var_59.force_grid_wrap == 0
    assert var_59.force_sort_within_sections is False
    assert var_59.lexicographical is False
    assert var_59.group_by_package is False
    assert var_59.ignore_whitespace is False
    assert f'{type(var_59.no_lines_before).__module__}.{type(var_59.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.no_lines_before) == 0
    assert var_59.no_inline_sort is False
    assert var_59.ignore_comments is False
    assert var_59.case_sensitive is False
    assert f'{type(var_59.sources).__module__}.{type(var_59.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_59.sources) == 1
    assert var_59.virtual_env == ''
    assert var_59.conda_env == ''
    assert var_59.ensure_newline_before_comments is False
    assert var_59.directory == '/workspace'
    assert var_59.profile == ''
    assert var_59.honor_noqa is False
    assert f'{type(var_59.src_paths).__module__}.{type(var_59.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_59.src_paths) == 2
    assert var_59.remove_redundant_aliases is False
    assert var_59.float_to_top is False
    assert var_59.filter_files is False
    assert var_59.formatter == ''
    assert var_59.formatting_function is None
    assert var_59.color_output is False
    assert f'{type(var_59.treat_comments_as_code).__module__}.{type(var_59.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.treat_comments_as_code) == 0
    assert var_59.treat_all_comments_as_code is False
    assert f'{type(var_59.supported_extensions).__module__}.{type(var_59.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.supported_extensions) == 4
    assert f'{type(var_59.blocked_extensions).__module__}.{type(var_59.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.blocked_extensions) == 1
    assert f'{type(var_59.constants).__module__}.{type(var_59.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.constants) == 0
    assert f'{type(var_59.classes).__module__}.{type(var_59.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.classes) == 0
    assert f'{type(var_59.variables).__module__}.{type(var_59.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.variables) == 0
    assert var_59.dedup_headings is False
    assert var_59.only_sections is False
    assert var_59.only_modified is False
    assert var_59.combine_straight_imports is False
    assert var_59.auto_identify_namespace_packages is True
    assert f'{type(var_59.namespace_packages).__module__}.{type(var_59.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_59.namespace_packages) == 0
    assert var_59.follow_links is True
    assert var_59.indented_import_headings is True
    assert var_59.honor_case_in_force_sorted_sections is False
    assert var_59.sort_relative_in_force_sorted_sections is False
    assert var_59.overwrite_in_place is False
    assert var_59.reverse_sort is False
    assert var_59.star_first is False
    assert var_59.git_ls_files == {}
    assert var_59.format_error == '{error}: {message}'
    assert var_59.format_success == '{success}: {message}'
    assert var_59.sort_order == 'natural'
    assert var_59.sort_reexports is False
    assert var_59.split_on_trailing_comma is False
    var_27.__repr__(var_6)

def test_case_23():
    var_0 = 'import os  # operating system\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.categorized_comments

def test_case_24():
    var_0 = 'from os import path,\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.trailing_commas

def test_case_25():
    var_0 = '# isort: imports-THIRDPARTY\nimport numpy\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.place_imports
    var_3 = len(var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 'import os\nimport sys\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\nfrom sys import argv\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = var_5.imports
    var_7 = len(var_6)
    var_8 = 'import os  # comment\nfrom sys import argv\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.categorized_comments
    var_11 = len(var_10)
    var_12 = ''
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'x = 1\ny = 2\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = var_15.lines_without_imports
    var_17 = len(var_16)
    var_18 = 'from os import \\\n    path\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = var_19.imports
    var_21 = len(var_20)
    var_22 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = 'import os as operating_system\nfrom sys import argv as args\n'
    var_25 = module_0.file_contents(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_25) == 14
    var_26 = 'straight'
    var_27 = var_25.as_map[var_26]
    var_28 = len(var_27)
    var_29 = 0
    var_30 = var_28 > var_29
    var_31 = 'from'
    var_32 = var_25.as_map[var_31]
    var_33 = len(var_32)
    var_34 = var_33 > var_29
    var_35 = 'import os\n'
    var_36 = module_0.file_contents(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    module_0.file_contents(var_11)

def test_case_27():
    var_0 = True
    var_1 = module_3.Config()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.settings.Config'
    assert var_1.py_version == 'py3'
    assert f'{type(var_1.force_to_top).__module__}.{type(var_1.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.force_to_top) == 0
    assert f'{type(var_1.skip).__module__}.{type(var_1.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip) == 19
    assert f'{type(var_1.extend_skip).__module__}.{type(var_1.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip) == 0
    assert f'{type(var_1.skip_glob).__module__}.{type(var_1.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.skip_glob) == 0
    assert f'{type(var_1.extend_skip_glob).__module__}.{type(var_1.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extend_skip_glob) == 0
    assert var_1.skip_gitignore is False
    assert var_1.line_length == 79
    assert var_1.wrap_length == 0
    assert var_1.line_ending == ''
    assert var_1.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_1.no_sections is False
    assert f'{type(var_1.known_future_library).__module__}.{type(var_1.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_future_library) == 1
    assert f'{type(var_1.known_third_party).__module__}.{type(var_1.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_third_party) == 0
    assert f'{type(var_1.known_first_party).__module__}.{type(var_1.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_first_party) == 0
    assert f'{type(var_1.known_local_folder).__module__}.{type(var_1.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_local_folder) == 0
    assert f'{type(var_1.known_standard_library).__module__}.{type(var_1.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.known_standard_library) == 234
    assert f'{type(var_1.extra_standard_library).__module__}.{type(var_1.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.extra_standard_library) == 0
    assert var_1.known_other == {}
    assert var_1.multi_line_output == module_1.WrapModes.GRID
    assert var_1.forced_separate == ()
    assert var_1.indent == '    '
    assert var_1.comment_prefix == '  #'
    assert var_1.length_sort is False
    assert var_1.length_sort_straight is False
    assert f'{type(var_1.length_sort_sections).__module__}.{type(var_1.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.length_sort_sections) == 0
    assert f'{type(var_1.add_imports).__module__}.{type(var_1.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.add_imports) == 0
    assert f'{type(var_1.remove_imports).__module__}.{type(var_1.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.remove_imports) == 0
    assert var_1.append_only is False
    assert var_1.reverse_relative is False
    assert var_1.force_single_line is False
    assert var_1.single_line_exclusions == ()
    assert var_1.default_section == 'THIRDPARTY'
    assert var_1.import_headings == {}
    assert var_1.import_footers == {}
    assert var_1.balanced_wrapping is False
    assert var_1.use_parentheses is False
    assert var_1.order_by_type is True
    assert var_1.atomic is False
    assert var_1.lines_before_imports == -1
    assert var_1.lines_after_imports == -1
    assert var_1.lines_between_sections == 1
    assert var_1.lines_between_types == 0
    assert var_1.combine_as_imports is False
    assert var_1.combine_star is False
    assert var_1.include_trailing_comma is False
    assert var_1.from_first is False
    assert var_1.verbose is False
    assert var_1.quiet is False
    assert var_1.force_adds is False
    assert var_1.force_alphabetical_sort_within_sections is False
    assert var_1.force_alphabetical_sort is False
    assert var_1.force_grid_wrap == 0
    assert var_1.force_sort_within_sections is False
    assert var_1.lexicographical is False
    assert var_1.group_by_package is False
    assert var_1.ignore_whitespace is False
    assert f'{type(var_1.no_lines_before).__module__}.{type(var_1.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.no_lines_before) == 0
    assert var_1.no_inline_sort is False
    assert var_1.ignore_comments is False
    assert var_1.case_sensitive is False
    assert f'{type(var_1.sources).__module__}.{type(var_1.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_1.sources) == 1
    assert var_1.virtual_env == ''
    assert var_1.conda_env == ''
    assert var_1.ensure_newline_before_comments is False
    assert var_1.directory == '/workspace'
    assert var_1.profile == ''
    assert var_1.honor_noqa is False
    assert f'{type(var_1.src_paths).__module__}.{type(var_1.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_1.src_paths) == 2
    assert var_1.remove_redundant_aliases is False
    assert var_1.float_to_top is False
    assert var_1.filter_files is False
    assert var_1.formatter == ''
    assert var_1.formatting_function is None
    assert var_1.color_output is False
    assert f'{type(var_1.treat_comments_as_code).__module__}.{type(var_1.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.treat_comments_as_code) == 0
    assert var_1.treat_all_comments_as_code is False
    assert f'{type(var_1.supported_extensions).__module__}.{type(var_1.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.supported_extensions) == 4
    assert f'{type(var_1.blocked_extensions).__module__}.{type(var_1.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.blocked_extensions) == 1
    assert f'{type(var_1.constants).__module__}.{type(var_1.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.constants) == 0
    assert f'{type(var_1.classes).__module__}.{type(var_1.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.classes) == 0
    assert f'{type(var_1.variables).__module__}.{type(var_1.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.variables) == 0
    assert var_1.dedup_headings is False
    assert var_1.only_sections is False
    assert var_1.only_modified is False
    assert var_1.combine_straight_imports is False
    assert var_1.auto_identify_namespace_packages is True
    assert f'{type(var_1.namespace_packages).__module__}.{type(var_1.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_1.namespace_packages) == 0
    assert var_1.follow_links is True
    assert var_1.indented_import_headings is True
    assert var_1.honor_case_in_force_sorted_sections is False
    assert var_1.sort_relative_in_force_sorted_sections is False
    assert var_1.overwrite_in_place is False
    assert var_1.reverse_sort is False
    assert var_1.star_first is False
    assert var_1.git_ls_files == {}
    assert var_1.format_error == '{error}: {message}'
    assert var_1.format_success == '{success}: {message}'
    assert var_1.sort_order == 'natural'
    assert var_1.sort_reexports is False
    assert var_1.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_2 = 'import os as os\n'
    var_3 = module_0.file_contents(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_3) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_4 = var_3.as_map

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = 'import numpy as np\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.as_map
    var_11 = str(var_10)
    var_12 = 'from os import path, getcwd\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = var_15.in_lines
    var_17 = len(var_16)
    var_18 = 'from os import \\\n    path, \\\n    getcwd\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = 'def hello():\n    pass\n'
    var_21 = module_0.file_contents(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_21) == 14
    var_22 = 'import os  # operating system\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = 'import os\n\ndef main():\n    pass\n'
    var_25 = module_0.file_contents(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_25) == 14
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_29) == 14
    var_30 = '\n'
    var_31 = module_0.file_contents(var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_31) == 14
    var_32 = 'from os impo>t path,\n'
    var_33 = module_0.file_contents(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_33) == 14
    var_34 = True
    var_35 = module_3.Config()
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'isort.settings.Config'
    assert var_35.py_version == 'py3'
    assert f'{type(var_35.force_to_top).__module__}.{type(var_35.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.force_to_top) == 0
    assert f'{type(var_35.skip).__module__}.{type(var_35.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.skip) == 19
    assert f'{type(var_35.extend_skip).__module__}.{type(var_35.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.extend_skip) == 0
    assert f'{type(var_35.skip_glob).__module__}.{type(var_35.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.skip_glob) == 0
    assert f'{type(var_35.extend_skip_glob).__module__}.{type(var_35.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.extend_skip_glob) == 0
    assert var_35.skip_gitignore is False
    assert var_35.line_length == 79
    assert var_35.wrap_length == 0
    assert var_35.line_ending == ''
    assert var_35.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_35.no_sections is False
    assert f'{type(var_35.known_future_library).__module__}.{type(var_35.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.known_future_library) == 1
    assert f'{type(var_35.known_third_party).__module__}.{type(var_35.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.known_third_party) == 0
    assert f'{type(var_35.known_first_party).__module__}.{type(var_35.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.known_first_party) == 0
    assert f'{type(var_35.known_local_folder).__module__}.{type(var_35.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.known_local_folder) == 0
    assert f'{type(var_35.known_standard_library).__module__}.{type(var_35.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.known_standard_library) == 234
    assert f'{type(var_35.extra_standard_library).__module__}.{type(var_35.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.extra_standard_library) == 0
    assert var_35.known_other == {}
    assert var_35.multi_line_output == module_1.WrapModes.GRID
    assert var_35.forced_separate == ()
    assert var_35.indent == '    '
    assert var_35.comment_prefix == '  #'
    assert var_35.length_sort is False
    assert var_35.length_sort_straight is False
    assert f'{type(var_35.length_sort_sections).__module__}.{type(var_35.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.length_sort_sections) == 0
    assert f'{type(var_35.add_imports).__module__}.{type(var_35.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.add_imports) == 0
    assert f'{type(var_35.remove_imports).__module__}.{type(var_35.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.remove_imports) == 0
    assert var_35.append_only is False
    assert var_35.reverse_relative is False
    assert var_35.force_single_line is False
    assert var_35.single_line_exclusions == ()
    assert var_35.default_section == 'THIRDPARTY'
    assert var_35.import_headings == {}
    assert var_35.import_footers == {}
    assert var_35.balanced_wrapping is False
    assert var_35.use_parentheses is False
    assert var_35.order_by_type is True
    assert var_35.atomic is False
    assert var_35.lines_before_imports == -1
    assert var_35.lines_after_imports == -1
    assert var_35.lines_between_sections == 1
    assert var_35.lines_between_types == 0
    assert var_35.combine_as_imports is False
    assert var_35.combine_star is False
    assert var_35.include_trailing_comma is False
    assert var_35.from_first is False
    assert var_35.verbose is False
    assert var_35.quiet is False
    assert var_35.force_adds is False
    assert var_35.force_alphabetical_sort_within_sections is False
    assert var_35.force_alphabetical_sort is False
    assert var_35.force_grid_wrap == 0
    assert var_35.force_sort_within_sections is False
    assert var_35.lexicographical is False
    assert var_35.group_by_package is False
    assert var_35.ignore_whitespace is False
    assert f'{type(var_35.no_lines_before).__module__}.{type(var_35.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.no_lines_before) == 0
    assert var_35.no_inline_sort is False
    assert var_35.ignore_comments is False
    assert var_35.case_sensitive is False
    assert f'{type(var_35.sources).__module__}.{type(var_35.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_35.sources) == 1
    assert var_35.virtual_env == ''
    assert var_35.conda_env == ''
    assert var_35.ensure_newline_before_comments is False
    assert var_35.directory == '/workspace'
    assert var_35.profile == ''
    assert var_35.honor_noqa is False
    assert f'{type(var_35.src_paths).__module__}.{type(var_35.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_35.src_paths) == 2
    assert var_35.remove_redundant_aliases is False
    assert var_35.float_to_top is False
    assert var_35.filter_files is False
    assert var_35.formatter == ''
    assert var_35.formatting_function is None
    assert var_35.color_output is False
    assert f'{type(var_35.treat_comments_as_code).__module__}.{type(var_35.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.treat_comments_as_code) == 0
    assert var_35.treat_all_comments_as_code is False
    assert f'{type(var_35.supported_extensions).__module__}.{type(var_35.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.supported_extensions) == 4
    assert f'{type(var_35.blocked_extensions).__module__}.{type(var_35.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.blocked_extensions) == 1
    assert f'{type(var_35.constants).__module__}.{type(var_35.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.constants) == 0
    assert f'{type(var_35.classes).__module__}.{type(var_35.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.classes) == 0
    assert f'{type(var_35.variables).__module__}.{type(var_35.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.variables) == 0
    assert var_35.dedup_headings is False
    assert var_35.only_sections is False
    assert var_35.only_modified is False
    assert var_35.combine_straight_imports is False
    assert var_35.auto_identify_namespace_packages is True
    assert f'{type(var_35.namespace_packages).__module__}.{type(var_35.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_35.namespace_packages) == 0
    assert var_35.follow_links is True
    assert var_35.indented_import_headings is True
    assert var_35.honor_case_in_force_sorted_sections is False
    assert var_35.sort_relative_in_force_sorted_sections is False
    assert var_35.overwrite_in_place is False
    assert var_35.reverse_sort is False
    assert var_35.star_first is False
    assert var_35.git_ls_files == {}
    assert var_35.format_error == '{error}: {message}'
    assert var_35.format_success == '{success}: {message}'
    assert var_35.sort_order == 'natural'
    assert var_35.sort_reexports is False
    assert var_35.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_36 = 'from os import path, getcwd\n'
    var_37 = module_0.file_contents(var_36, var_35)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_37) == 14
    var_38 = module_3.Config()
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.settings.Config'
    assert var_38.py_version == 'py3'
    assert f'{type(var_38.force_to_top).__module__}.{type(var_38.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.force_to_top) == 0
    assert f'{type(var_38.skip).__module__}.{type(var_38.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.skip) == 19
    assert f'{type(var_38.extend_skip).__module__}.{type(var_38.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.extend_skip) == 0
    assert f'{type(var_38.skip_glob).__module__}.{type(var_38.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.skip_glob) == 0
    assert f'{type(var_38.extend_skip_glob).__module__}.{type(var_38.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.extend_skip_glob) == 0
    assert var_38.skip_gitignore is False
    assert var_38.line_length == 79
    assert var_38.wrap_length == 0
    assert var_38.line_ending == ''
    assert var_38.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_38.no_sections is False
    assert f'{type(var_38.known_future_library).__module__}.{type(var_38.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.known_future_library) == 1
    assert f'{type(var_38.known_third_party).__module__}.{type(var_38.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.known_third_party) == 0
    assert f'{type(var_38.known_first_party).__module__}.{type(var_38.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.known_first_party) == 0
    assert f'{type(var_38.known_local_folder).__module__}.{type(var_38.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.known_local_folder) == 0
    assert f'{type(var_38.known_standard_library).__module__}.{type(var_38.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.known_standard_library) == 234
    assert f'{type(var_38.extra_standard_library).__module__}.{type(var_38.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.extra_standard_library) == 0
    assert var_38.known_other == {}
    assert var_38.multi_line_output == module_1.WrapModes.GRID
    assert var_38.forced_separate == ()
    assert var_38.indent == '    '
    assert var_38.comment_prefix == '  #'
    assert var_38.length_sort is False
    assert var_38.length_sort_straight is False
    assert f'{type(var_38.length_sort_sections).__module__}.{type(var_38.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.length_sort_sections) == 0
    assert f'{type(var_38.add_imports).__module__}.{type(var_38.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.add_imports) == 0
    assert f'{type(var_38.remove_imports).__module__}.{type(var_38.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.remove_imports) == 0
    assert var_38.append_only is False
    assert var_38.reverse_relative is False
    assert var_38.force_single_line is False
    assert var_38.single_line_exclusions == ()
    assert var_38.default_section == 'THIRDPARTY'
    assert var_38.import_headings == {}
    assert var_38.import_footers == {}
    assert var_38.balanced_wrapping is False
    assert var_38.use_parentheses is False
    assert var_38.order_by_type is True
    assert var_38.atomic is False
    assert var_38.lines_before_imports == -1
    assert var_38.lines_after_imports == -1
    assert var_38.lines_between_sections == 1
    assert var_38.lines_between_types == 0
    assert var_38.combine_as_imports is False
    assert var_38.combine_star is False
    assert var_38.include_trailing_comma is False
    assert var_38.from_first is False
    assert var_38.verbose is False
    assert var_38.quiet is False
    assert var_38.force_adds is False
    assert var_38.force_alphabetical_sort_within_sections is False
    assert var_38.force_alphabetical_sort is False
    assert var_38.force_grid_wrap == 0
    assert var_38.force_sort_within_sections is False
    assert var_38.lexicographical is False
    assert var_38.group_by_package is False
    assert var_38.ignore_whitespace is False
    assert f'{type(var_38.no_lines_before).__module__}.{type(var_38.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.no_lines_before) == 0
    assert var_38.no_inline_sort is False
    assert var_38.ignore_comments is False
    assert var_38.case_sensitive is False
    assert f'{type(var_38.sources).__module__}.{type(var_38.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_38.sources) == 1
    assert var_38.virtual_env == ''
    assert var_38.conda_env == ''
    assert var_38.ensure_newline_before_comments is False
    assert var_38.directory == '/workspace'
    assert var_38.profile == ''
    assert var_38.honor_noqa is False
    assert f'{type(var_38.src_paths).__module__}.{type(var_38.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_38.src_paths) == 2
    assert var_38.remove_redundant_aliases is False
    assert var_38.float_to_top is False
    assert var_38.filter_files is False
    assert var_38.formatter == ''
    assert var_38.formatting_function is None
    assert var_38.color_output is False
    assert f'{type(var_38.treat_comments_as_code).__module__}.{type(var_38.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.treat_comments_as_code) == 0
    assert var_38.treat_all_comments_as_code is False
    assert f'{type(var_38.supported_extensions).__module__}.{type(var_38.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.supported_extensions) == 4
    assert f'{type(var_38.blocked_extensions).__module__}.{type(var_38.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.blocked_extensions) == 1
    assert f'{type(var_38.constants).__module__}.{type(var_38.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.constants) == 0
    assert f'{type(var_38.classes).__module__}.{type(var_38.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.classes) == 0
    assert f'{type(var_38.variables).__module__}.{type(var_38.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.variables) == 0
    assert var_38.dedup_headings is False
    assert var_38.only_sections is False
    assert var_38.only_modified is False
    assert var_38.combine_straight_imports is False
    assert var_38.auto_identify_namespace_packages is True
    assert f'{type(var_38.namespace_packages).__module__}.{type(var_38.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_38.namespace_packages) == 0
    assert var_38.follow_links is True
    assert var_38.indented_import_headings is True
    assert var_38.honor_case_in_force_sorted_sections is False
    assert var_38.sort_relative_in_force_sorted_sections is False
    assert var_38.overwrite_in_place is False
    assert var_38.reverse_sort is False
    assert var_38.star_first is False
    assert var_38.git_ls_files == {}
    assert var_38.format_error == '{error}: {message}'
    assert var_38.format_success == '{success}: {message}'
    assert var_38.sort_order == 'natural'
    assert var_38.sort_reexports is False
    assert var_38.split_on_trailing_comma is False
    var_39 = '# isort: split\nimport os\n'
    var_40 = module_0.file_contents(var_39, var_38)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_40) == 14
    var_41 = var_40.in_lines
    var_42 = len(var_41)
    var_43 = 'from libc.stdlib cimport malloc, free\n'
    var_44 = module_0.file_contents(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_44) == 14
    var_45 = 'import os; import sys\n'
    var_46 = module_0.file_contents(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_46) == 14
    var_47 = 'from os import (\n    path,  # path module\n    getcwd\n)\n'
    var_48 = module_0.file_contents(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_48) == 14
    var_49 = var_23.categorized_comments
    var_50 = 'import os\n'
    var_51 = module_0.file_contents(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_51) == 14
    var_52 = 'in_lines'
    var_53 = hasattr(var_51, var_52)
    var_54 = 'lines_without_imports'
    var_55 = hasattr(var_51, var_54)
    var_56 = 'import_index'
    var_57 = hasattr(var_51, var_56)
    var_58 = 'imports'
    var_59 = hasattr(var_51, var_58)
    var_60 = 'as_map'
    var_61 = hasattr(var_51, var_60)
    var_62 = 'categorized_comments'
    var_63 = hasattr(var_51, var_62)
    var_64 = 'trailing_commas'
    var_65 = hasattr(var_51, var_64)
    var_66 = 'verbose_output'
    var_67 = hasattr(var_51, var_66)
    var_27.__setitem__(var_42, var_61)

def test_case_29():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = 'import os\nimport ys;from pathlib import Path\n'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = 'import numpy as np\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.as_map
    var_11 = str(var_10)
    var_12 = 'from os import path, getcwd\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = var_15.in_lines
    var_17 = len(var_16)
    var_18 = 'from os import \\\n    path, \\\n    getcwd\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = 'def h[llo():\n    pass\n'
    var_21 = module_0.file_contents(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_21) == 14
    var_22 = 'import os  # operating system\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = 'import os\n\ndef main():\n    pass\n'
    var_25 = module_0.file_contents(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_25) == 14
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_29) == 14
    var_30 = module_0.file_contents(var_11)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_30) == 14
    var_31 = 'from os import path,\n'
    var_32 = module_0.file_contents(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_32) == 14
    var_33 = True
    var_34 = module_3.Config()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'isort.settings.Config'
    assert var_34.py_version == 'py3'
    assert f'{type(var_34.force_to_top).__module__}.{type(var_34.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.force_to_top) == 0
    assert f'{type(var_34.skip).__module__}.{type(var_34.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.skip) == 19
    assert f'{type(var_34.extend_skip).__module__}.{type(var_34.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extend_skip) == 0
    assert f'{type(var_34.skip_glob).__module__}.{type(var_34.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.skip_glob) == 0
    assert f'{type(var_34.extend_skip_glob).__module__}.{type(var_34.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extend_skip_glob) == 0
    assert var_34.skip_gitignore is False
    assert var_34.line_length == 79
    assert var_34.wrap_length == 0
    assert var_34.line_ending == ''
    assert var_34.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_34.no_sections is False
    assert f'{type(var_34.known_future_library).__module__}.{type(var_34.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_future_library) == 1
    assert f'{type(var_34.known_third_party).__module__}.{type(var_34.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_third_party) == 0
    assert f'{type(var_34.known_first_party).__module__}.{type(var_34.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_first_party) == 0
    assert f'{type(var_34.known_local_folder).__module__}.{type(var_34.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_local_folder) == 0
    assert f'{type(var_34.known_standard_library).__module__}.{type(var_34.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_standard_library) == 234
    assert f'{type(var_34.extra_standard_library).__module__}.{type(var_34.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extra_standard_library) == 0
    assert var_34.known_other == {}
    assert var_34.multi_line_output == module_1.WrapModes.GRID
    assert var_34.forced_separate == ()
    assert var_34.indent == '    '
    assert var_34.comment_prefix == '  #'
    assert var_34.length_sort is False
    assert var_34.length_sort_straight is False
    assert f'{type(var_34.length_sort_sections).__module__}.{type(var_34.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.length_sort_sections) == 0
    assert f'{type(var_34.add_imports).__module__}.{type(var_34.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.add_imports) == 0
    assert f'{type(var_34.remove_imports).__module__}.{type(var_34.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.remove_imports) == 0
    assert var_34.append_only is False
    assert var_34.reverse_relative is False
    assert var_34.force_single_line is False
    assert var_34.single_line_exclusions == ()
    assert var_34.default_section == 'THIRDPARTY'
    assert var_34.import_headings == {}
    assert var_34.import_footers == {}
    assert var_34.balanced_wrapping is False
    assert var_34.use_parentheses is False
    assert var_34.order_by_type is True
    assert var_34.atomic is False
    assert var_34.lines_before_imports == -1
    assert var_34.lines_after_imports == -1
    assert var_34.lines_between_sections == 1
    assert var_34.lines_between_types == 0
    assert var_34.combine_as_imports is False
    assert var_34.combine_star is False
    assert var_34.include_trailing_comma is False
    assert var_34.from_first is False
    assert var_34.verbose is False
    assert var_34.quiet is False
    assert var_34.force_adds is False
    assert var_34.force_alphabetical_sort_within_sections is False
    assert var_34.force_alphabetical_sort is False
    assert var_34.force_grid_wrap == 0
    assert var_34.force_sort_within_sections is False
    assert var_34.lexicographical is False
    assert var_34.group_by_package is False
    assert var_34.ignore_whitespace is False
    assert f'{type(var_34.no_lines_before).__module__}.{type(var_34.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.no_lines_before) == 0
    assert var_34.no_inline_sort is False
    assert var_34.ignore_comments is False
    assert var_34.case_sensitive is False
    assert f'{type(var_34.sources).__module__}.{type(var_34.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_34.sources) == 1
    assert var_34.virtual_env == ''
    assert var_34.conda_env == ''
    assert var_34.ensure_newline_before_comments is False
    assert var_34.directory == '/workspace'
    assert var_34.profile == ''
    assert var_34.honor_noqa is False
    assert f'{type(var_34.src_paths).__module__}.{type(var_34.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_34.src_paths) == 2
    assert var_34.remove_redundant_aliases is False
    assert var_34.float_to_top is False
    assert var_34.filter_files is False
    assert var_34.formatter == ''
    assert var_34.formatting_function is None
    assert var_34.color_output is False
    assert f'{type(var_34.treat_comments_as_code).__module__}.{type(var_34.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.treat_comments_as_code) == 0
    assert var_34.treat_all_comments_as_code is False
    assert f'{type(var_34.supported_extensions).__module__}.{type(var_34.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.supported_extensions) == 4
    assert f'{type(var_34.blocked_extensions).__module__}.{type(var_34.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.blocked_extensions) == 1
    assert f'{type(var_34.constants).__module__}.{type(var_34.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.constants) == 0
    assert f'{type(var_34.classes).__module__}.{type(var_34.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.classes) == 0
    assert f'{type(var_34.variables).__module__}.{type(var_34.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.variables) == 0
    assert var_34.dedup_headings is False
    assert var_34.only_sections is False
    assert var_34.only_modified is False
    assert var_34.combine_straight_imports is False
    assert var_34.auto_identify_namespace_packages is True
    assert f'{type(var_34.namespace_packages).__module__}.{type(var_34.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.namespace_packages) == 0
    assert var_34.follow_links is True
    assert var_34.indented_import_headings is True
    assert var_34.honor_case_in_force_sorted_sections is False
    assert var_34.sort_relative_in_force_sorted_sections is False
    assert var_34.overwrite_in_place is False
    assert var_34.reverse_sort is False
    assert var_34.star_first is False
    assert var_34.git_ls_files == {}
    assert var_34.format_error == '{error}: {message}'
    assert var_34.format_success == '{success}: {message}'
    assert var_34.sort_order == 'natural'
    assert var_34.sort_reexports is False
    assert var_34.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_35 = 'from os import path, getcwd\n'
    var_36 = module_0.file_contents(var_35, var_34)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = module_3.Config()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'isort.settings.Config'
    assert var_37.py_version == 'py3'
    assert f'{type(var_37.force_to_top).__module__}.{type(var_37.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.force_to_top) == 0
    assert f'{type(var_37.skip).__module__}.{type(var_37.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.skip) == 19
    assert f'{type(var_37.extend_skip).__module__}.{type(var_37.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extend_skip) == 0
    assert f'{type(var_37.skip_glob).__module__}.{type(var_37.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.skip_glob) == 0
    assert f'{type(var_37.extend_skip_glob).__module__}.{type(var_37.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extend_skip_glob) == 0
    assert var_37.skip_gitignore is False
    assert var_37.line_length == 79
    assert var_37.wrap_length == 0
    assert var_37.line_ending == ''
    assert var_37.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_37.no_sections is False
    assert f'{type(var_37.known_future_library).__module__}.{type(var_37.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_future_library) == 1
    assert f'{type(var_37.known_third_party).__module__}.{type(var_37.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_third_party) == 0
    assert f'{type(var_37.known_first_party).__module__}.{type(var_37.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_first_party) == 0
    assert f'{type(var_37.known_local_folder).__module__}.{type(var_37.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_local_folder) == 0
    assert f'{type(var_37.known_standard_library).__module__}.{type(var_37.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_standard_library) == 234
    assert f'{type(var_37.extra_standard_library).__module__}.{type(var_37.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extra_standard_library) == 0
    assert var_37.known_other == {}
    assert var_37.multi_line_output == module_1.WrapModes.GRID
    assert var_37.forced_separate == ()
    assert var_37.indent == '    '
    assert var_37.comment_prefix == '  #'
    assert var_37.length_sort is False
    assert var_37.length_sort_straight is False
    assert f'{type(var_37.length_sort_sections).__module__}.{type(var_37.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.length_sort_sections) == 0
    assert f'{type(var_37.add_imports).__module__}.{type(var_37.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.add_imports) == 0
    assert f'{type(var_37.remove_imports).__module__}.{type(var_37.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.remove_imports) == 0
    assert var_37.append_only is False
    assert var_37.reverse_relative is False
    assert var_37.force_single_line is False
    assert var_37.single_line_exclusions == ()
    assert var_37.default_section == 'THIRDPARTY'
    assert var_37.import_headings == {}
    assert var_37.import_footers == {}
    assert var_37.balanced_wrapping is False
    assert var_37.use_parentheses is False
    assert var_37.order_by_type is True
    assert var_37.atomic is False
    assert var_37.lines_before_imports == -1
    assert var_37.lines_after_imports == -1
    assert var_37.lines_between_sections == 1
    assert var_37.lines_between_types == 0
    assert var_37.combine_as_imports is False
    assert var_37.combine_star is False
    assert var_37.include_trailing_comma is False
    assert var_37.from_first is False
    assert var_37.verbose is False
    assert var_37.quiet is False
    assert var_37.force_adds is False
    assert var_37.force_alphabetical_sort_within_sections is False
    assert var_37.force_alphabetical_sort is False
    assert var_37.force_grid_wrap == 0
    assert var_37.force_sort_within_sections is False
    assert var_37.lexicographical is False
    assert var_37.group_by_package is False
    assert var_37.ignore_whitespace is False
    assert f'{type(var_37.no_lines_before).__module__}.{type(var_37.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.no_lines_before) == 0
    assert var_37.no_inline_sort is False
    assert var_37.ignore_comments is False
    assert var_37.case_sensitive is False
    assert f'{type(var_37.sources).__module__}.{type(var_37.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_37.sources) == 1
    assert var_37.virtual_env == ''
    assert var_37.conda_env == ''
    assert var_37.ensure_newline_before_comments is False
    assert var_37.directory == '/workspace'
    assert var_37.profile == ''
    assert var_37.honor_noqa is False
    assert f'{type(var_37.src_paths).__module__}.{type(var_37.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_37.src_paths) == 2
    assert var_37.remove_redundant_aliases is False
    assert var_37.float_to_top is False
    assert var_37.filter_files is False
    assert var_37.formatter == ''
    assert var_37.formatting_function is None
    assert var_37.color_output is False
    assert f'{type(var_37.treat_comments_as_code).__module__}.{type(var_37.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.treat_comments_as_code) == 0
    assert var_37.treat_all_comments_as_code is False
    assert f'{type(var_37.supported_extensions).__module__}.{type(var_37.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.supported_extensions) == 4
    assert f'{type(var_37.blocked_extensions).__module__}.{type(var_37.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.blocked_extensions) == 1
    assert f'{type(var_37.constants).__module__}.{type(var_37.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.constants) == 0
    assert f'{type(var_37.classes).__module__}.{type(var_37.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.classes) == 0
    assert f'{type(var_37.variables).__module__}.{type(var_37.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.variables) == 0
    assert var_37.dedup_headings is False
    assert var_37.only_sections is False
    assert var_37.only_modified is False
    assert var_37.combine_straight_imports is False
    assert var_37.auto_identify_namespace_packages is True
    assert f'{type(var_37.namespace_packages).__module__}.{type(var_37.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.namespace_packages) == 0
    assert var_37.follow_links is True
    assert var_37.indented_import_headings is True
    assert var_37.honor_case_in_force_sorted_sections is False
    assert var_37.sort_relative_in_force_sorted_sections is False
    assert var_37.overwrite_in_place is False
    assert var_37.reverse_sort is False
    assert var_37.star_first is False
    assert var_37.git_ls_files == {}
    assert var_37.format_error == '{error}: {message}'
    assert var_37.format_success == '{success}: {message}'
    assert var_37.sort_order == 'natural'
    assert var_37.sort_reexports is False
    assert var_37.split_on_trailing_comma is False
    var_38 = '# isort: split\nimport os\n'
    var_39 = module_0.file_contents(var_38, var_37)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_39) == 14
    var_40 = var_39.in_lines
    var_41 = len(var_2)
    var_42 = 'from libc.stdlib cimport malloc, free\n'
    var_43 = module_0.file_contents(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_43) == 14
    var_44 = 'import os; import sys\n'
    var_45 = module_0.file_contents(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_45) == 14
    var_46 = 'from os import (\n    path,  # path module\n    getcwd\n)\n'
    var_47 = module_0.file_contents(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_47) == 14
    var_48 = var_47.categorized_comments
    var_49 = 'import os\n'
    var_50 = module_0.file_contents(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_50) == 14
    var_51 = 'lines_without_imports'
    var_52 = hasattr(var_50, var_51)
    var_53 = 'import_index'
    var_54 = hasattr(var_50, var_53)
    var_55 = 'imports'
    var_56 = hasattr(var_50, var_55)
    var_57 = 'as_map'
    var_58 = hasattr(var_50, var_57)
    var_59 = 'categorized_comments'
    var_60 = hasattr(var_50, var_59)
    var_61 = 'trailing_commas'
    var_62 = hasattr(var_50, var_61)
    var_63 = 'verbose_output'
    var_64 = hasattr(var_50, var_63)
    var_65 = module_3.Config()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'isort.settings.Config'
    assert var_65.py_version == 'py3'
    assert f'{type(var_65.force_to_top).__module__}.{type(var_65.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.force_to_top) == 0
    assert f'{type(var_65.skip).__module__}.{type(var_65.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.skip) == 19
    assert f'{type(var_65.extend_skip).__module__}.{type(var_65.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extend_skip) == 0
    assert f'{type(var_65.skip_glob).__module__}.{type(var_65.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.skip_glob) == 0
    assert f'{type(var_65.extend_skip_glob).__module__}.{type(var_65.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extend_skip_glob) == 0
    assert var_65.skip_gitignore is False
    assert var_65.line_length == 79
    assert var_65.wrap_length == 0
    assert var_65.line_ending == ''
    assert var_65.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_65.no_sections is False
    assert f'{type(var_65.known_future_library).__module__}.{type(var_65.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_future_library) == 1
    assert f'{type(var_65.known_third_party).__module__}.{type(var_65.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_third_party) == 0
    assert f'{type(var_65.known_first_party).__module__}.{type(var_65.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_first_party) == 0
    assert f'{type(var_65.known_local_folder).__module__}.{type(var_65.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_local_folder) == 0
    assert f'{type(var_65.known_standard_library).__module__}.{type(var_65.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_standard_library) == 234
    assert f'{type(var_65.extra_standard_library).__module__}.{type(var_65.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extra_standard_library) == 0
    assert var_65.known_other == {}
    assert var_65.multi_line_output == module_1.WrapModes.GRID
    assert var_65.forced_separate == ()
    assert var_65.indent == '    '
    assert var_65.comment_prefix == '  #'
    assert var_65.length_sort is False
    assert var_65.length_sort_straight is False
    assert f'{type(var_65.length_sort_sections).__module__}.{type(var_65.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.length_sort_sections) == 0
    assert f'{type(var_65.add_imports).__module__}.{type(var_65.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.add_imports) == 0
    assert f'{type(var_65.remove_imports).__module__}.{type(var_65.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.remove_imports) == 0
    assert var_65.append_only is False
    assert var_65.reverse_relative is False
    assert var_65.force_single_line is False
    assert var_65.single_line_exclusions == ()
    assert var_65.default_section == 'THIRDPARTY'
    assert var_65.import_headings == {}
    assert var_65.import_footers == {}
    assert var_65.balanced_wrapping is False
    assert var_65.use_parentheses is False
    assert var_65.order_by_type is True
    assert var_65.atomic is False
    assert var_65.lines_before_imports == -1
    assert var_65.lines_after_imports == -1
    assert var_65.lines_between_sections == 1
    assert var_65.lines_between_types == 0
    assert var_65.combine_as_imports is False
    assert var_65.combine_star is False
    assert var_65.include_trailing_comma is False
    assert var_65.from_first is False
    assert var_65.verbose is False
    assert var_65.quiet is False
    assert var_65.force_adds is False
    assert var_65.force_alphabetical_sort_within_sections is False
    assert var_65.force_alphabetical_sort is False
    assert var_65.force_grid_wrap == 0
    assert var_65.force_sort_within_sections is False
    assert var_65.lexicographical is False
    assert var_65.group_by_package is False
    assert var_65.ignore_whitespace is False
    assert f'{type(var_65.no_lines_before).__module__}.{type(var_65.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.no_lines_before) == 0
    assert var_65.no_inline_sort is False
    assert var_65.ignore_comments is False
    assert var_65.case_sensitive is False
    assert f'{type(var_65.sources).__module__}.{type(var_65.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_65.sources) == 1
    assert var_65.virtual_env == ''
    assert var_65.conda_env == ''
    assert var_65.ensure_newline_before_comments is False
    assert var_65.directory == '/workspace'
    assert var_65.profile == ''
    assert var_65.honor_noqa is False
    assert f'{type(var_65.src_paths).__module__}.{type(var_65.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_65.src_paths) == 2
    assert var_65.remove_redundant_aliases is False
    assert var_65.float_to_top is False
    assert var_65.filter_files is False
    assert var_65.formatter == ''
    assert var_65.formatting_function is None
    assert var_65.color_output is False
    assert f'{type(var_65.treat_comments_as_code).__module__}.{type(var_65.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.treat_comments_as_code) == 0
    assert var_65.treat_all_comments_as_code is False
    assert f'{type(var_65.supported_extensions).__module__}.{type(var_65.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.supported_extensions) == 4
    assert f'{type(var_65.blocked_extensions).__module__}.{type(var_65.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.blocked_extensions) == 1
    assert f'{type(var_65.constants).__module__}.{type(var_65.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.constants) == 0
    assert f'{type(var_65.classes).__module__}.{type(var_65.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.classes) == 0
    assert f'{type(var_65.variables).__module__}.{type(var_65.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.variables) == 0
    assert var_65.dedup_headings is False
    assert var_65.only_sections is False
    assert var_65.only_modified is False
    assert var_65.combine_straight_imports is False
    assert var_65.auto_identify_namespace_packages is True
    assert f'{type(var_65.namespace_packages).__module__}.{type(var_65.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.namespace_packages) == 0
    assert var_65.follow_links is True
    assert var_65.indented_import_headings is True
    assert var_65.honor_case_in_force_sorted_sections is False
    assert var_65.sort_relative_in_force_sorted_sections is False
    assert var_65.overwrite_in_place is False
    assert var_65.reverse_sort is False
    assert var_65.star_first is False
    assert var_65.git_ls_files == {}
    assert var_65.format_error == '{error}: {message}'
    assert var_65.format_success == '{success}: {message}'
    assert var_65.sort_order == 'natural'
    assert var_65.sort_reexports is False
    assert var_65.split_on_trailing_comma is False
    var_66 = 'import os as os\n'
    var_67 = module_0.file_contents(var_66, var_65)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_67) == 14

def test_case_30():
    var_0 = 'import os\n'
    var_1 = module_0.file_contents(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_1) == 14
    assert module_0.TYPE_CHECKING is False
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
    assert f'{type(module_0.ParsedContent.in_lines).__module__}.{type(module_0.ParsedContent.in_lines).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.lines_without_imports).__module__}.{type(module_0.ParsedContent.lines_without_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_index).__module__}.{type(module_0.ParsedContent.import_index).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.place_imports).__module__}.{type(module_0.ParsedContent.place_imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.import_placements).__module__}.{type(module_0.ParsedContent.import_placements).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.as_map).__module__}.{type(module_0.ParsedContent.as_map).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.imports).__module__}.{type(module_0.ParsedContent.imports).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.categorized_comments).__module__}.{type(module_0.ParsedContent.categorized_comments).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.change_count).__module__}.{type(module_0.ParsedContent.change_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.original_line_count).__module__}.{type(module_0.ParsedContent.original_line_count).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.line_separator).__module__}.{type(module_0.ParsedContent.line_separator).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.sections).__module__}.{type(module_0.ParsedContent.sections).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.verbose_output).__module__}.{type(module_0.ParsedContent.verbose_output).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.ParsedContent.trailing_commas).__module__}.{type(module_0.ParsedContent.trailing_commas).__qualname__}' == '_collections._tuplegetter'
    var_2 = var_1.imports
    var_3 = len(var_2)
    var_4 = 'from os import path\n'
    var_5 = module_0.file_contents(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = 'import os\nimport sys\nfrom pathlib import Path\n'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = 'import numpy as np\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.as_map
    var_11 = str(var_10)
    var_12 = 'from os import path, getcwd\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = var_15.in_lines
    var_17 = len(var_16)
    var_18 = 'from os import \\\n    path, \\\n    g\\tcwd\n'
    var_19 = module_0.file_contents(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_19) == 14
    var_20 = 'def h[llo():\n    pass\n'
    var_21 = module_0.file_contents(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_21) == 14
    var_22 = 'import os  # operating system\n'
    var_23 = module_0.file_contents(var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_23) == 14
    var_24 = 'import os\n\ndef main():\n    pass\n'
    var_25 = module_0.file_contents(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_25) == 14
    var_26 = var_25.lines_without_imports
    var_27 = len(var_26)
    var_28 = ''
    var_29 = module_0.file_contents(var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_29) == 14
    var_30 = module_0.file_contents(var_11)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_30) == 14
    var_31 = 'from os import path,\n'
    var_32 = module_0.file_contents(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_32) == 14
    var_33 = True
    var_34 = module_3.Config()
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'isort.settings.Config'
    assert var_34.py_version == 'py3'
    assert f'{type(var_34.force_to_top).__module__}.{type(var_34.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.force_to_top) == 0
    assert f'{type(var_34.skip).__module__}.{type(var_34.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.skip) == 19
    assert f'{type(var_34.extend_skip).__module__}.{type(var_34.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extend_skip) == 0
    assert f'{type(var_34.skip_glob).__module__}.{type(var_34.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.skip_glob) == 0
    assert f'{type(var_34.extend_skip_glob).__module__}.{type(var_34.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extend_skip_glob) == 0
    assert var_34.skip_gitignore is False
    assert var_34.line_length == 79
    assert var_34.wrap_length == 0
    assert var_34.line_ending == ''
    assert var_34.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_34.no_sections is False
    assert f'{type(var_34.known_future_library).__module__}.{type(var_34.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_future_library) == 1
    assert f'{type(var_34.known_third_party).__module__}.{type(var_34.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_third_party) == 0
    assert f'{type(var_34.known_first_party).__module__}.{type(var_34.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_first_party) == 0
    assert f'{type(var_34.known_local_folder).__module__}.{type(var_34.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_local_folder) == 0
    assert f'{type(var_34.known_standard_library).__module__}.{type(var_34.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.known_standard_library) == 234
    assert f'{type(var_34.extra_standard_library).__module__}.{type(var_34.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.extra_standard_library) == 0
    assert var_34.known_other == {}
    assert var_34.multi_line_output == module_1.WrapModes.GRID
    assert var_34.forced_separate == ()
    assert var_34.indent == '    '
    assert var_34.comment_prefix == '  #'
    assert var_34.length_sort is False
    assert var_34.length_sort_straight is False
    assert f'{type(var_34.length_sort_sections).__module__}.{type(var_34.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.length_sort_sections) == 0
    assert f'{type(var_34.add_imports).__module__}.{type(var_34.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.add_imports) == 0
    assert f'{type(var_34.remove_imports).__module__}.{type(var_34.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.remove_imports) == 0
    assert var_34.append_only is False
    assert var_34.reverse_relative is False
    assert var_34.force_single_line is False
    assert var_34.single_line_exclusions == ()
    assert var_34.default_section == 'THIRDPARTY'
    assert var_34.import_headings == {}
    assert var_34.import_footers == {}
    assert var_34.balanced_wrapping is False
    assert var_34.use_parentheses is False
    assert var_34.order_by_type is True
    assert var_34.atomic is False
    assert var_34.lines_before_imports == -1
    assert var_34.lines_after_imports == -1
    assert var_34.lines_between_sections == 1
    assert var_34.lines_between_types == 0
    assert var_34.combine_as_imports is False
    assert var_34.combine_star is False
    assert var_34.include_trailing_comma is False
    assert var_34.from_first is False
    assert var_34.verbose is False
    assert var_34.quiet is False
    assert var_34.force_adds is False
    assert var_34.force_alphabetical_sort_within_sections is False
    assert var_34.force_alphabetical_sort is False
    assert var_34.force_grid_wrap == 0
    assert var_34.force_sort_within_sections is False
    assert var_34.lexicographical is False
    assert var_34.group_by_package is False
    assert var_34.ignore_whitespace is False
    assert f'{type(var_34.no_lines_before).__module__}.{type(var_34.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.no_lines_before) == 0
    assert var_34.no_inline_sort is False
    assert var_34.ignore_comments is False
    assert var_34.case_sensitive is False
    assert f'{type(var_34.sources).__module__}.{type(var_34.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_34.sources) == 1
    assert var_34.virtual_env == ''
    assert var_34.conda_env == ''
    assert var_34.ensure_newline_before_comments is False
    assert var_34.directory == '/workspace'
    assert var_34.profile == ''
    assert var_34.honor_noqa is False
    assert f'{type(var_34.src_paths).__module__}.{type(var_34.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_34.src_paths) == 2
    assert var_34.remove_redundant_aliases is False
    assert var_34.float_to_top is False
    assert var_34.filter_files is False
    assert var_34.formatter == ''
    assert var_34.formatting_function is None
    assert var_34.color_output is False
    assert f'{type(var_34.treat_comments_as_code).__module__}.{type(var_34.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.treat_comments_as_code) == 0
    assert var_34.treat_all_comments_as_code is False
    assert f'{type(var_34.supported_extensions).__module__}.{type(var_34.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.supported_extensions) == 4
    assert f'{type(var_34.blocked_extensions).__module__}.{type(var_34.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.blocked_extensions) == 1
    assert f'{type(var_34.constants).__module__}.{type(var_34.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.constants) == 0
    assert f'{type(var_34.classes).__module__}.{type(var_34.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.classes) == 0
    assert f'{type(var_34.variables).__module__}.{type(var_34.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.variables) == 0
    assert var_34.dedup_headings is False
    assert var_34.only_sections is False
    assert var_34.only_modified is False
    assert var_34.combine_straight_imports is False
    assert var_34.auto_identify_namespace_packages is True
    assert f'{type(var_34.namespace_packages).__module__}.{type(var_34.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_34.namespace_packages) == 0
    assert var_34.follow_links is True
    assert var_34.indented_import_headings is True
    assert var_34.honor_case_in_force_sorted_sections is False
    assert var_34.sort_relative_in_force_sorted_sections is False
    assert var_34.overwrite_in_place is False
    assert var_34.reverse_sort is False
    assert var_34.star_first is False
    assert var_34.git_ls_files == {}
    assert var_34.format_error == '{error}: {message}'
    assert var_34.format_success == '{success}: {message}'
    assert var_34.sort_order == 'natural'
    assert var_34.sort_reexports is False
    assert var_34.split_on_trailing_comma is False
    assert module_3.TYPE_CHECKING is False
    assert module_3.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_3.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.FIRSTPARTY == 'FIRSTPARTY'
    assert module_3.FUTURE == 'FUTURE'
    assert module_3.LOCALFOLDER == 'LOCALFOLDER'
    assert module_3.STDLIB == 'STDLIB'
    assert module_3.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_3.CYTHON_EXTENSIONS).__module__}.{type(module_3.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_3.SUPPORTED_EXTENSIONS).__module__}.{type(module_3.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_3.BLOCKED_EXTENSIONS).__module__}.{type(module_3.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.BLOCKED_EXTENSIONS) == 1
    assert module_3.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_3.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_3.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_3.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_3.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_3.DEFAULT_SKIP).__module__}.{type(module_3.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_SKIP) == 19
    assert module_3.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_3.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_3.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_3.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_3.KNOWN_PREFIX == 'known_'
    assert module_3.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_3.RUNTIME_SOURCE == 'runtime'
    assert module_3.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_3.DEFAULT_CONFIG).__module__}.{type(module_3.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_3.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_3.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_3.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip).__module__}.{type(module_3.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_3.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_3.DEFAULT_CONFIG.skip_gitignore is False
    assert module_3.DEFAULT_CONFIG.line_length == 79
    assert module_3.DEFAULT_CONFIG.wrap_length == 0
    assert module_3.DEFAULT_CONFIG.line_ending == ''
    assert module_3.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_3.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_3.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_3.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_3.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_3.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_3.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_3.DEFAULT_CONFIG.known_other == {}
    assert module_3.DEFAULT_CONFIG.multi_line_output == module_1.WrapModes.GRID
    assert module_3.DEFAULT_CONFIG.forced_separate == ()
    assert module_3.DEFAULT_CONFIG.indent == '    '
    assert module_3.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_3.DEFAULT_CONFIG.length_sort is False
    assert module_3.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_3.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_3.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.add_imports).__module__}.{type(module_3.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_3.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.remove_imports) == 0
    assert module_3.DEFAULT_CONFIG.append_only is False
    assert module_3.DEFAULT_CONFIG.reverse_relative is False
    assert module_3.DEFAULT_CONFIG.force_single_line is False
    assert module_3.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_3.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_3.DEFAULT_CONFIG.import_headings == {}
    assert module_3.DEFAULT_CONFIG.import_footers == {}
    assert module_3.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_3.DEFAULT_CONFIG.use_parentheses is False
    assert module_3.DEFAULT_CONFIG.order_by_type is True
    assert module_3.DEFAULT_CONFIG.atomic is False
    assert module_3.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_3.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_3.DEFAULT_CONFIG.lines_between_types == 0
    assert module_3.DEFAULT_CONFIG.combine_as_imports is False
    assert module_3.DEFAULT_CONFIG.combine_star is False
    assert module_3.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_3.DEFAULT_CONFIG.from_first is False
    assert module_3.DEFAULT_CONFIG.verbose is False
    assert module_3.DEFAULT_CONFIG.quiet is False
    assert module_3.DEFAULT_CONFIG.force_adds is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_3.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_3.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_3.DEFAULT_CONFIG.lexicographical is False
    assert module_3.DEFAULT_CONFIG.group_by_package is False
    assert module_3.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_3.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_3.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_3.DEFAULT_CONFIG.no_inline_sort is False
    assert module_3.DEFAULT_CONFIG.ignore_comments is False
    assert module_3.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_3.DEFAULT_CONFIG.sources).__module__}.{type(module_3.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.sources) == 1
    assert module_3.DEFAULT_CONFIG.virtual_env == ''
    assert module_3.DEFAULT_CONFIG.conda_env == ''
    assert module_3.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_3.DEFAULT_CONFIG.directory == '/workspace'
    assert module_3.DEFAULT_CONFIG.profile == ''
    assert module_3.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_3.DEFAULT_CONFIG.src_paths).__module__}.{type(module_3.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_3.DEFAULT_CONFIG.src_paths) == 2
    assert module_3.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_3.DEFAULT_CONFIG.float_to_top is False
    assert module_3.DEFAULT_CONFIG.filter_files is False
    assert module_3.DEFAULT_CONFIG.formatter == ''
    assert module_3.DEFAULT_CONFIG.formatting_function is None
    assert module_3.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_3.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_3.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_3.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_3.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_3.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_3.DEFAULT_CONFIG.constants).__module__}.{type(module_3.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.classes).__module__}.{type(module_3.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_3.DEFAULT_CONFIG.variables).__module__}.{type(module_3.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.variables) == 0
    assert module_3.DEFAULT_CONFIG.dedup_headings is False
    assert module_3.DEFAULT_CONFIG.only_sections is False
    assert module_3.DEFAULT_CONFIG.only_modified is False
    assert module_3.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_3.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_3.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_3.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_3.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_3.DEFAULT_CONFIG.follow_links is True
    assert module_3.DEFAULT_CONFIG.indented_import_headings is True
    assert module_3.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_3.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_3.DEFAULT_CONFIG.reverse_sort is False
    assert module_3.DEFAULT_CONFIG.star_first is False
    assert module_3.DEFAULT_CONFIG.git_ls_files == {}
    assert module_3.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_3.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_3.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_3.DEFAULT_CONFIG.sort_reexports is False
    assert module_3.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_3.Config.known_patterns).__module__}.{type(module_3.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments).__module__}.{type(module_3.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.section_comments_end).__module__}.{type(module_3.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skips).__module__}.{type(module_3.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.skip_globs).__module__}.{type(module_3.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_3.Config.sorting_function).__module__}.{type(module_3.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_35 = '.\x0cw \x0birKeuB:Vk'
    var_36 = module_0.file_contents(var_35, var_34)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = module_3.Config()
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'isort.settings.Config'
    assert var_37.py_version == 'py3'
    assert f'{type(var_37.force_to_top).__module__}.{type(var_37.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.force_to_top) == 0
    assert f'{type(var_37.skip).__module__}.{type(var_37.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.skip) == 19
    assert f'{type(var_37.extend_skip).__module__}.{type(var_37.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extend_skip) == 0
    assert f'{type(var_37.skip_glob).__module__}.{type(var_37.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.skip_glob) == 0
    assert f'{type(var_37.extend_skip_glob).__module__}.{type(var_37.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extend_skip_glob) == 0
    assert var_37.skip_gitignore is False
    assert var_37.line_length == 79
    assert var_37.wrap_length == 0
    assert var_37.line_ending == ''
    assert var_37.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_37.no_sections is False
    assert f'{type(var_37.known_future_library).__module__}.{type(var_37.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_future_library) == 1
    assert f'{type(var_37.known_third_party).__module__}.{type(var_37.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_third_party) == 0
    assert f'{type(var_37.known_first_party).__module__}.{type(var_37.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_first_party) == 0
    assert f'{type(var_37.known_local_folder).__module__}.{type(var_37.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_local_folder) == 0
    assert f'{type(var_37.known_standard_library).__module__}.{type(var_37.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.known_standard_library) == 234
    assert f'{type(var_37.extra_standard_library).__module__}.{type(var_37.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.extra_standard_library) == 0
    assert var_37.known_other == {}
    assert var_37.multi_line_output == module_1.WrapModes.GRID
    assert var_37.forced_separate == ()
    assert var_37.indent == '    '
    assert var_37.comment_prefix == '  #'
    assert var_37.length_sort is False
    assert var_37.length_sort_straight is False
    assert f'{type(var_37.length_sort_sections).__module__}.{type(var_37.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.length_sort_sections) == 0
    assert f'{type(var_37.add_imports).__module__}.{type(var_37.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.add_imports) == 0
    assert f'{type(var_37.remove_imports).__module__}.{type(var_37.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.remove_imports) == 0
    assert var_37.append_only is False
    assert var_37.reverse_relative is False
    assert var_37.force_single_line is False
    assert var_37.single_line_exclusions == ()
    assert var_37.default_section == 'THIRDPARTY'
    assert var_37.import_headings == {}
    assert var_37.import_footers == {}
    assert var_37.balanced_wrapping is False
    assert var_37.use_parentheses is False
    assert var_37.order_by_type is True
    assert var_37.atomic is False
    assert var_37.lines_before_imports == -1
    assert var_37.lines_after_imports == -1
    assert var_37.lines_between_sections == 1
    assert var_37.lines_between_types == 0
    assert var_37.combine_as_imports is False
    assert var_37.combine_star is False
    assert var_37.include_trailing_comma is False
    assert var_37.from_first is False
    assert var_37.verbose is False
    assert var_37.quiet is False
    assert var_37.force_adds is False
    assert var_37.force_alphabetical_sort_within_sections is False
    assert var_37.force_alphabetical_sort is False
    assert var_37.force_grid_wrap == 0
    assert var_37.force_sort_within_sections is False
    assert var_37.lexicographical is False
    assert var_37.group_by_package is False
    assert var_37.ignore_whitespace is False
    assert f'{type(var_37.no_lines_before).__module__}.{type(var_37.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.no_lines_before) == 0
    assert var_37.no_inline_sort is False
    assert var_37.ignore_comments is False
    assert var_37.case_sensitive is False
    assert f'{type(var_37.sources).__module__}.{type(var_37.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_37.sources) == 1
    assert var_37.virtual_env == ''
    assert var_37.conda_env == ''
    assert var_37.ensure_newline_before_comments is False
    assert var_37.directory == '/workspace'
    assert var_37.profile == ''
    assert var_37.honor_noqa is False
    assert f'{type(var_37.src_paths).__module__}.{type(var_37.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_37.src_paths) == 2
    assert var_37.remove_redundant_aliases is False
    assert var_37.float_to_top is False
    assert var_37.filter_files is False
    assert var_37.formatter == ''
    assert var_37.formatting_function is None
    assert var_37.color_output is False
    assert f'{type(var_37.treat_comments_as_code).__module__}.{type(var_37.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.treat_comments_as_code) == 0
    assert var_37.treat_all_comments_as_code is False
    assert f'{type(var_37.supported_extensions).__module__}.{type(var_37.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.supported_extensions) == 4
    assert f'{type(var_37.blocked_extensions).__module__}.{type(var_37.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.blocked_extensions) == 1
    assert f'{type(var_37.constants).__module__}.{type(var_37.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.constants) == 0
    assert f'{type(var_37.classes).__module__}.{type(var_37.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.classes) == 0
    assert f'{type(var_37.variables).__module__}.{type(var_37.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.variables) == 0
    assert var_37.dedup_headings is False
    assert var_37.only_sections is False
    assert var_37.only_modified is False
    assert var_37.combine_straight_imports is False
    assert var_37.auto_identify_namespace_packages is True
    assert f'{type(var_37.namespace_packages).__module__}.{type(var_37.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_37.namespace_packages) == 0
    assert var_37.follow_links is True
    assert var_37.indented_import_headings is True
    assert var_37.honor_case_in_force_sorted_sections is False
    assert var_37.sort_relative_in_force_sorted_sections is False
    assert var_37.overwrite_in_place is False
    assert var_37.reverse_sort is False
    assert var_37.star_first is False
    assert var_37.git_ls_files == {}
    assert var_37.format_error == '{error}: {message}'
    assert var_37.format_success == '{success}: {message}'
    assert var_37.sort_order == 'natural'
    assert var_37.sort_reexports is False
    assert var_37.split_on_trailing_comma is False
    var_38 = '# isort: split\nimport os\n'
    var_39 = module_0.file_contents(var_38, var_37)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_39) == 14
    var_40 = var_39.in_lines
    var_41 = len(var_2)
    var_42 = 'from libc.stdlib cimport malloc, free\n'
    var_43 = module_0.file_contents(var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_43) == 14
    var_44 = 'import os; import sys\n'
    var_45 = module_0.file_contents(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_45) == 14
    var_46 = 'from os import (\n    path,  # path module\n    getcwd\n)\n'
    var_47 = module_0.file_contents(var_46)
    assert f'{type(var_47).__module__}.{type(var_47).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_47) == 14
    var_48 = var_47.categorized_comments
    var_49 = 'import os\n'
    var_50 = module_0.file_contents(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_50) == 14
    var_51 = 'lines_without_imports'
    var_52 = hasattr(var_50, var_51)
    var_53 = 'import_index'
    var_54 = hasattr(var_50, var_53)
    var_55 = 'imports'
    var_56 = hasattr(var_50, var_55)
    var_57 = 'as_map'
    var_58 = hasattr(var_50, var_57)
    var_59 = 'categorized_comments'
    var_60 = hasattr(var_50, var_59)
    var_61 = 'trailing_commas'
    var_62 = hasattr(var_50, var_61)
    var_63 = 'verbose_output'
    var_64 = hasattr(var_50, var_63)
    var_65 = module_3.Config()
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'isort.settings.Config'
    assert var_65.py_version == 'py3'
    assert f'{type(var_65.force_to_top).__module__}.{type(var_65.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.force_to_top) == 0
    assert f'{type(var_65.skip).__module__}.{type(var_65.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.skip) == 19
    assert f'{type(var_65.extend_skip).__module__}.{type(var_65.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extend_skip) == 0
    assert f'{type(var_65.skip_glob).__module__}.{type(var_65.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.skip_glob) == 0
    assert f'{type(var_65.extend_skip_glob).__module__}.{type(var_65.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extend_skip_glob) == 0
    assert var_65.skip_gitignore is False
    assert var_65.line_length == 79
    assert var_65.wrap_length == 0
    assert var_65.line_ending == ''
    assert var_65.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_65.no_sections is False
    assert f'{type(var_65.known_future_library).__module__}.{type(var_65.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_future_library) == 1
    assert f'{type(var_65.known_third_party).__module__}.{type(var_65.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_third_party) == 0
    assert f'{type(var_65.known_first_party).__module__}.{type(var_65.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_first_party) == 0
    assert f'{type(var_65.known_local_folder).__module__}.{type(var_65.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_local_folder) == 0
    assert f'{type(var_65.known_standard_library).__module__}.{type(var_65.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.known_standard_library) == 234
    assert f'{type(var_65.extra_standard_library).__module__}.{type(var_65.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.extra_standard_library) == 0
    assert var_65.known_other == {}
    assert var_65.multi_line_output == module_1.WrapModes.GRID
    assert var_65.forced_separate == ()
    assert var_65.indent == '    '
    assert var_65.comment_prefix == '  #'
    assert var_65.length_sort is False
    assert var_65.length_sort_straight is False
    assert f'{type(var_65.length_sort_sections).__module__}.{type(var_65.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.length_sort_sections) == 0
    assert f'{type(var_65.add_imports).__module__}.{type(var_65.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.add_imports) == 0
    assert f'{type(var_65.remove_imports).__module__}.{type(var_65.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.remove_imports) == 0
    assert var_65.append_only is False
    assert var_65.reverse_relative is False
    assert var_65.force_single_line is False
    assert var_65.single_line_exclusions == ()
    assert var_65.default_section == 'THIRDPARTY'
    assert var_65.import_headings == {}
    assert var_65.import_footers == {}
    assert var_65.balanced_wrapping is False
    assert var_65.use_parentheses is False
    assert var_65.order_by_type is True
    assert var_65.atomic is False
    assert var_65.lines_before_imports == -1
    assert var_65.lines_after_imports == -1
    assert var_65.lines_between_sections == 1
    assert var_65.lines_between_types == 0
    assert var_65.combine_as_imports is False
    assert var_65.combine_star is False
    assert var_65.include_trailing_comma is False
    assert var_65.from_first is False
    assert var_65.verbose is False
    assert var_65.quiet is False
    assert var_65.force_adds is False
    assert var_65.force_alphabetical_sort_within_sections is False
    assert var_65.force_alphabetical_sort is False
    assert var_65.force_grid_wrap == 0
    assert var_65.force_sort_within_sections is False
    assert var_65.lexicographical is False
    assert var_65.group_by_package is False
    assert var_65.ignore_whitespace is False
    assert f'{type(var_65.no_lines_before).__module__}.{type(var_65.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.no_lines_before) == 0
    assert var_65.no_inline_sort is False
    assert var_65.ignore_comments is False
    assert var_65.case_sensitive is False
    assert f'{type(var_65.sources).__module__}.{type(var_65.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_65.sources) == 1
    assert var_65.virtual_env == ''
    assert var_65.conda_env == ''
    assert var_65.ensure_newline_before_comments is False
    assert var_65.directory == '/workspace'
    assert var_65.profile == ''
    assert var_65.honor_noqa is False
    assert f'{type(var_65.src_paths).__module__}.{type(var_65.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_65.src_paths) == 2
    assert var_65.remove_redundant_aliases is False
    assert var_65.float_to_top is False
    assert var_65.filter_files is False
    assert var_65.formatter == ''
    assert var_65.formatting_function is None
    assert var_65.color_output is False
    assert f'{type(var_65.treat_comments_as_code).__module__}.{type(var_65.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.treat_comments_as_code) == 0
    assert var_65.treat_all_comments_as_code is False
    assert f'{type(var_65.supported_extensions).__module__}.{type(var_65.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.supported_extensions) == 4
    assert f'{type(var_65.blocked_extensions).__module__}.{type(var_65.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.blocked_extensions) == 1
    assert f'{type(var_65.constants).__module__}.{type(var_65.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.constants) == 0
    assert f'{type(var_65.classes).__module__}.{type(var_65.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.classes) == 0
    assert f'{type(var_65.variables).__module__}.{type(var_65.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.variables) == 0
    assert var_65.dedup_headings is False
    assert var_65.only_sections is False
    assert var_65.only_modified is False
    assert var_65.combine_straight_imports is False
    assert var_65.auto_identify_namespace_packages is True
    assert f'{type(var_65.namespace_packages).__module__}.{type(var_65.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_65.namespace_packages) == 0
    assert var_65.follow_links is True
    assert var_65.indented_import_headings is True
    assert var_65.honor_case_in_force_sorted_sections is False
    assert var_65.sort_relative_in_force_sorted_sections is False
    assert var_65.overwrite_in_place is False
    assert var_65.reverse_sort is False
    assert var_65.star_first is False
    assert var_65.git_ls_files == {}
    assert var_65.format_error == '{error}: {message}'
    assert var_65.format_success == '{success}: {message}'
    assert var_65.sort_order == 'natural'
    assert var_65.sort_reexports is False
    assert var_65.split_on_trailing_comma is False
    var_66 = 'import os as os\n'
    var_67 = module_0.file_contents(var_66, var_65)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_67) == 14