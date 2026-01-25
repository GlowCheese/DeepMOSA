# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.parse as module_0
import isort.wrap_modes as module_1
import re as module_2
import isort.settings as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.import_type(var_0)

def test_case_1():
    var_0 = 'Qq@d2Ct['
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

def test_case_2():
    var_0 = '~d'
    var_1 = module_0.strip_syntax(var_0)
    assert var_1 == '~d'
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

def test_case_3():
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

def test_case_4():
    var_0 = 'mRskb&:z\x0b\x0bLc~7*%I'
    var_1 = module_0.normalize_line(var_0)
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

def test_case_5():
    var_0 = 'L5`&StZ\x0bMI. \r'
    var_1 = '3%\x0bL$$V!5'
    var_2 = 's;woe'
    var_3 = module_0.file_contents(var_0)
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
    var_4 = module_0.strip_syntax(var_2)
    assert var_4 == 's;woe'
    var_5 = module_0.file_contents(var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_5) == 14
    var_6 = module_0.file_contents(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_6) == 14

def test_case_6():
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

def test_case_7():
    var_0 = None
    var_1 = 'UAa\\C-5LYa"Hoj'
    var_2 = module_0.skip_line(var_1, var_1, var_0, var_0)
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
    var_3 = '-(+mM@MK'
    var_4 = module_0.skip_line(var_3, var_0, var_0, var_0)

def test_case_8():
    var_0 = '~:Cs+b{&~F\x0b\nN*T\nYC'
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

def test_case_9():
    var_0 = '"""Module docstring."""\nimport os\n'
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

def test_case_10():
    var_0 = '-(+mM@MK'
    var_1 = module_0.normalize_line(var_0)
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
    var_2 = 'L5`&StZ\x0bMI. \r'
    var_3 = '3%\x0bL$$V!5'
    var_4 = 's;woe'
    var_5 = module_0.file_contents(var_2)
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
    var_6 = module_0.strip_syntax(var_4)
    assert var_6 == 's;woe'
    var_7 = module_0.file_contents(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = module_0.file_contents(var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_8) == 14
    var_9 = "azh'px~=9~92#=3--P\x0b"
    var_10 = None
    var_11 = module_0.skip_line(var_9, var_3, var_10, var_1, var_10)
    var_12 = 'KP>\x0bagK?'
    var_13 = module_0.strip_syntax(var_12)
    assert var_13 == 'KP> agK?'

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '-(+mM@MK'
    var_1 = module_0.normalize_line(var_0)
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
    var_2 = 'A~&\\;mJ56"JdSX'
    var_3 = None
    var_4 = module_0.skip_line(var_2, var_0, var_3, var_1)
    var_5 = False
    var_6 = '3%\x0bL$$V!5'
    var_7 = 's;woe'
    var_8 = module_0.strip_syntax(var_7)
    assert var_8 == 's;woe'
    var_9 = module_0.file_contents(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
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
    var_10 = module_0.file_contents(var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_10) == 14
    var_11 = '^dBuV\\0 yp^'
    var_12 = module_0.import_type(var_0)
    var_13 = module_0.skip_line(var_8, var_11, var_5, var_3, var_3)
    module_0.strip_syntax(var_3)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '-(+mM@MK'
    var_1 = module_0.normalize_line(var_0)
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
    var_2 = 'A~&\\;mJ56"JdSX'
    var_3 = 'R>YoK(f)g~e!~"x$'
    var_4 = None
    var_5 = module_0.skip_line(var_2, var_3, var_4, var_1)
    var_6 = False
    var_7 = '3%\x0bL$$V!5'
    var_8 = 's;woe'
    var_9 = module_0.strip_syntax(var_8)
    assert var_9 == 's;woe'
    var_10 = module_2.purge()
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
    var_11 = module_0.file_contents(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_11) == 14
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
    var_12 = ''
    var_13 = module_0.import_type(var_3)
    var_14 = module_0.skip_line(var_3, var_12, var_6, var_4, var_4)
    var_15 = None
    var_16 = ';XPU6'
    var_17 = ''
    var_18 = module_0.skip_line(var_16, var_17, var_4, var_1)
    module_0.file_contents(var_15)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'A~&\\;mJ56"JdSX'
    var_1 = None
    var_2 = 'kka*Cp_k'
    var_3 = module_0.file_contents(var_2)
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
    var_4 = ''
    var_5 = 's;woe'
    var_6 = module_0.strip_syntax(var_5)
    assert var_6 == 's;woe'
    var_7 = module_0.file_contents(var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = module_0.file_contents(var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_8) == 14
    var_9 = "azh'px~=9~92#=3--P\x0b"
    var_10 = "~C?gd'"
    var_11 = False
    var_12 = module_0.skip_line(var_10, var_1, var_1, var_1, var_11)
    var_13 = (var_9, var_0)
    module_0.skip_line(var_1, var_1, var_1, var_13)

def test_case_14():
    var_0 = '@_k'
    var_1 = module_0.normalize_line(var_0)
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
    var_2 = 'A~&\\;mJ56"JdSX'
    var_3 = 'R>YoK(f)g~e!~"x$'
    var_4 = None
    var_5 = module_0.skip_line(var_2, var_3, var_4, var_1)
    var_6 = 'kka*Cp_k'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
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
    var_8 = '3%\x0bL$$V!5'
    var_9 = 's;woe'
    var_10 = module_0.strip_syntax(var_9)
    assert var_10 == 's;woe'
    var_11 = module_0.file_contents(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_11) == 14
    var_12 = module_0.file_contents(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_12) == 14
    var_13 = module_0.skip_line(var_2, var_4, var_4, var_1)
    var_14 = "~C?gd'"
    var_15 = module_0.skip_line(var_14, var_4, var_4, var_1)
    var_16 = '#:d" @{B&EQhty'
    var_17 = module_0.file_contents(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_17) == 14

def test_case_15():
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
    var_12 = 'import numpy as np\nfrom os import path as p\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'straight'
    var_15 = var_13.as_map[var_14]
    var_16 = len(var_15)
    var_17 = 0
    var_18 = var_16 > var_17
    var_19 = 'from os impo>t (\n    path,\n    getcwd\n)\n'
    var_20 = module_0.file_contents(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_20) == 14
    var_21 = 'from os import path, \\\n    getcwd\n'
    var_22 = module_0.file_contents(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_22) == 14
    var_23 = 'x = 1\ny = 2\n'
    var_24 = module_0.file_contents(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_24) == 14
    var_25 = var_24.lines_without_imports
    var_26 = len(var_25)
    var_27 = 'import os  # isort:skip\nimport sys\n'
    var_28 = module_0.file_contents(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_28) == 14
    var_29 = var_28.lines_without_imports
    var_30 = len(var_29)
    var_31 = 'import os\r\nimport sys\r\n'
    var_32 = module_0.file_contents(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_32) == 14
    var_33 = ''
    var_34 = module_0.file_contents(var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_34) == 14
    var_35 = 'import os\n'
    var_36 = module_0.file_contents(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = 'from os import (\n    path,\n    getcwd,\n)\n'
    var_38 = module_0.file_contents(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_38) == 14
    with pytest.raises(AttributeError):
        var_39 = var_30.trailing_commas

def test_case_16():
    var_0 = 'import os'
    var_1 = module_0.import_type(var_0)
    assert var_1 == 'straight'
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
    var_2 = 'import sys'
    var_3 = module_0.import_type(var_2)
    assert var_3 == 'straight'
    var_4 = 'import numpy as np'
    var_5 = module_0.import_type(var_4)
    assert var_5 == 'straight'
    var_6 = 'cimport numpy'
    var_7 = module_0.import_type(var_6)
    assert var_7 == 'straight'
    var_8 = 'cimport cython'
    var_9 = module_0.import_type(var_8)
    assert var_9 == 'straight'
    var_10 = 'from os import path'
    var_11 = module_0.import_type(var_10)
    assert var_11 == 'from'
    var_12 = 'from ..module import func'
    var_13 = module_0.import_type(var_12)
    assert var_13 == 'from'
    var_14 = 'from typing import List, Dict'
    var_15 = module_0.import_type(var_14)
    assert var_15 == 'from'
    var_16 = 'import os  # isort:skip'
    var_17 = module_0.import_type(var_16)
    assert var_17 is None
    var_18 = 'import sys  # isort: skip'
    var_19 = module_0.import_type(var_18)
    assert var_19 is None
    var_20 = 'from os import path  # isort:skip'
    var_21 = module_0.import_type(var_20)
    assert var_21 is None
    var_22 = 'import os  # isort: split'
    var_23 = module_0.import_type(var_22)
    assert var_23 is None
    var_24 = 'from os import path  # isort: split'
    var_25 = module_0.import_type(var_24)
    assert var_25 is None
    var_26 = 'x = 5'
    var_27 = module_0.import_type(var_26)
    assert var_27 is None
    var_28 = 'def func():'
    var_29 = module_0.import_type(var_28)
    assert var_29 is None
    var_30 = "print('hello')"
    var_31 = module_0.import_type(var_30)
    assert var_31 is None
    var_32 = ''
    var_33 = module_0.import_type(var_32)
    assert var_33 is None
    var_34 = '# comment'
    var_35 = module_0.import_type(var_34)
    assert var_35 is None
    var_36 = True
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
    var_38 = 'import os  # noqa'
    var_39 = module_0.import_type(var_38, var_37)
    assert var_39 == 'straight'
    var_40 = 'import os  # NOQA'
    var_41 = module_0.import_type(var_40, var_37)
    assert var_41 == 'straight'
    var_42 = 'from os import path  # noqa'
    var_43 = module_0.import_type(var_42, var_37)
    assert var_43 == 'from'
    var_44 = False
    var_45 = module_3.Config()
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'isort.settings.Config'
    assert var_45.py_version == 'py3'
    assert f'{type(var_45.force_to_top).__module__}.{type(var_45.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.force_to_top) == 0
    assert f'{type(var_45.skip).__module__}.{type(var_45.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.skip) == 19
    assert f'{type(var_45.extend_skip).__module__}.{type(var_45.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.extend_skip) == 0
    assert f'{type(var_45.skip_glob).__module__}.{type(var_45.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.skip_glob) == 0
    assert f'{type(var_45.extend_skip_glob).__module__}.{type(var_45.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.extend_skip_glob) == 0
    assert var_45.skip_gitignore is False
    assert var_45.line_length == 79
    assert var_45.wrap_length == 0
    assert var_45.line_ending == ''
    assert var_45.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_45.no_sections is False
    assert f'{type(var_45.known_future_library).__module__}.{type(var_45.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.known_future_library) == 1
    assert f'{type(var_45.known_third_party).__module__}.{type(var_45.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.known_third_party) == 0
    assert f'{type(var_45.known_first_party).__module__}.{type(var_45.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.known_first_party) == 0
    assert f'{type(var_45.known_local_folder).__module__}.{type(var_45.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.known_local_folder) == 0
    assert f'{type(var_45.known_standard_library).__module__}.{type(var_45.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.known_standard_library) == 234
    assert f'{type(var_45.extra_standard_library).__module__}.{type(var_45.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.extra_standard_library) == 0
    assert var_45.known_other == {}
    assert var_45.multi_line_output == module_1.WrapModes.GRID
    assert var_45.forced_separate == ()
    assert var_45.indent == '    '
    assert var_45.comment_prefix == '  #'
    assert var_45.length_sort is False
    assert var_45.length_sort_straight is False
    assert f'{type(var_45.length_sort_sections).__module__}.{type(var_45.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.length_sort_sections) == 0
    assert f'{type(var_45.add_imports).__module__}.{type(var_45.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.add_imports) == 0
    assert f'{type(var_45.remove_imports).__module__}.{type(var_45.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.remove_imports) == 0
    assert var_45.append_only is False
    assert var_45.reverse_relative is False
    assert var_45.force_single_line is False
    assert var_45.single_line_exclusions == ()
    assert var_45.default_section == 'THIRDPARTY'
    assert var_45.import_headings == {}
    assert var_45.import_footers == {}
    assert var_45.balanced_wrapping is False
    assert var_45.use_parentheses is False
    assert var_45.order_by_type is True
    assert var_45.atomic is False
    assert var_45.lines_before_imports == -1
    assert var_45.lines_after_imports == -1
    assert var_45.lines_between_sections == 1
    assert var_45.lines_between_types == 0
    assert var_45.combine_as_imports is False
    assert var_45.combine_star is False
    assert var_45.include_trailing_comma is False
    assert var_45.from_first is False
    assert var_45.verbose is False
    assert var_45.quiet is False
    assert var_45.force_adds is False
    assert var_45.force_alphabetical_sort_within_sections is False
    assert var_45.force_alphabetical_sort is False
    assert var_45.force_grid_wrap == 0
    assert var_45.force_sort_within_sections is False
    assert var_45.lexicographical is False
    assert var_45.group_by_package is False
    assert var_45.ignore_whitespace is False
    assert f'{type(var_45.no_lines_before).__module__}.{type(var_45.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.no_lines_before) == 0
    assert var_45.no_inline_sort is False
    assert var_45.ignore_comments is False
    assert var_45.case_sensitive is False
    assert f'{type(var_45.sources).__module__}.{type(var_45.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_45.sources) == 1
    assert var_45.virtual_env == ''
    assert var_45.conda_env == ''
    assert var_45.ensure_newline_before_comments is False
    assert var_45.directory == '/workspace'
    assert var_45.profile == ''
    assert var_45.honor_noqa is False
    assert f'{type(var_45.src_paths).__module__}.{type(var_45.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_45.src_paths) == 2
    assert var_45.remove_redundant_aliases is False
    assert var_45.float_to_top is False
    assert var_45.filter_files is False
    assert var_45.formatter == ''
    assert var_45.formatting_function is None
    assert var_45.color_output is False
    assert f'{type(var_45.treat_comments_as_code).__module__}.{type(var_45.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.treat_comments_as_code) == 0
    assert var_45.treat_all_comments_as_code is False
    assert f'{type(var_45.supported_extensions).__module__}.{type(var_45.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.supported_extensions) == 4
    assert f'{type(var_45.blocked_extensions).__module__}.{type(var_45.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.blocked_extensions) == 1
    assert f'{type(var_45.constants).__module__}.{type(var_45.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.constants) == 0
    assert f'{type(var_45.classes).__module__}.{type(var_45.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.classes) == 0
    assert f'{type(var_45.variables).__module__}.{type(var_45.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.variables) == 0
    assert var_45.dedup_headings is False
    assert var_45.only_sections is False
    assert var_45.only_modified is False
    assert var_45.combine_straight_imports is False
    assert var_45.auto_identify_namespace_packages is True
    assert f'{type(var_45.namespace_packages).__module__}.{type(var_45.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_45.namespace_packages) == 0
    assert var_45.follow_links is True
    assert var_45.indented_import_headings is True
    assert var_45.honor_case_in_force_sorted_sections is False
    assert var_45.sort_relative_in_force_sorted_sections is False
    assert var_45.overwrite_in_place is False
    assert var_45.reverse_sort is False
    assert var_45.star_first is False
    assert var_45.git_ls_files == {}
    assert var_45.format_error == '{error}: {message}'
    assert var_45.format_success == '{success}: {message}'
    assert var_45.sort_order == 'natural'
    assert var_45.sort_reexports is False
    assert var_45.split_on_trailing_comma is False
    var_46 = module_0.import_type(var_38, var_45)
    assert var_46 == 'straight'
    var_47 = module_0.import_type(var_42, var_45)
    assert var_47 == 'from'
    var_48 = '  import os'
    var_49 = module_0.import_type(var_48)
    assert var_49 is None
    var_50 = 'importlib'
    var_51 = module_0.import_type(var_50)
    assert var_51 is None
    var_52 = 'from_module import something'
    var_53 = module_0.import_type(var_52)
    assert var_53 is None
    var_54 = 'import'
    var_55 = module_0.import_type(var_54)
    assert var_55 is None
    var_56 = 'from'
    var_57 = module_0.import_type(var_56)
    assert var_57 is None

def test_case_17():
    var_0 = 'from os import path\nfrom sys import argv\n'
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
    var_0 = 'from os import \\\n    path, \\\n    getcwd\n'
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
    var_0 = 'from os import (\n    path,\n    getcwd\n)\n'
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

def test_case_21():
    var_0 = 'Test the file_contents function with various import scenarios.'
    var_1 = 'import os\nimport sys\n'
    var_2 = module_0.file_contents(var_1)
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
    var_3 = var_2.imports
    var_4 = module_0.file_contents(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_4) == 14
    var_5 = var_4.imports
    var_6 = 'import os  # comment\nfrom sys import argv\n'
    var_7 = module_0.file_contents(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_7) == 14
    var_8 = var_7.categorized_comments
    var_9 = len(var_8)
    var_10 = 'import numpy as np\nfrom os import path as p\n'
    var_11 = module_0.file_contents(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_11) == 14
    var_12 = 'straight'
    var_13 = var_11.as_map[var_12]
    var_14 = len(var_13)
    var_15 = 0
    var_16 = var_14 > var_15
    var_17 = 'from os import path, \\\n    getcwd\n'
    var_18 = module_0.file_contents(var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_18) == 14
    var_19 = 'x = 1\ny = 2\n'
    var_20 = module_0.file_contents(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_20) == 14
    var_21 = var_20.lines_without_imports
    var_22 = len(var_21)
    var_23 = 'import os  # isort:skip\nimport sys\n'
    var_24 = module_0.file_contents(var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_24) == 14
    var_25 = var_24.lines_without_imports
    var_26 = module_0.file_contents(var_0)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_26) == 14
    var_27 = 'import os\n'
    var_28 = module_0.file_contents(var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_28) == 14
    var_29 = module_0.file_contents(var_23)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_29) == 14
    var_30 = var_29.trailing_commas
    var_31 = len(var_30)
    var_32 = 'import os; import sys\n'
    var_33 = module_0.file_contents(var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_33) == 14
    var_34 = 'from os import path, getcwd, chdir\n'
    var_35 = module_0.file_contents(var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_35) == 14
    var_36 = 'import os\nimport sys\n'
    var_37 = module_0.file_contents(var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_37) == 14
    var_38 = var_37.lines_without_imports
    var_39 = len(var_38)
    var_40 = var_37.original_line_count
    var_41 = module_3.Config()
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'isort.settings.Config'
    assert var_41.py_version == 'py3'
    assert f'{type(var_41.force_to_top).__module__}.{type(var_41.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.force_to_top) == 0
    assert f'{type(var_41.skip).__module__}.{type(var_41.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.skip) == 19
    assert f'{type(var_41.extend_skip).__module__}.{type(var_41.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.extend_skip) == 0
    assert f'{type(var_41.skip_glob).__module__}.{type(var_41.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.skip_glob) == 0
    assert f'{type(var_41.extend_skip_glob).__module__}.{type(var_41.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.extend_skip_glob) == 0
    assert var_41.skip_gitignore is False
    assert var_41.line_length == 79
    assert var_41.wrap_length == 0
    assert var_41.line_ending == ''
    assert var_41.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_41.no_sections is False
    assert f'{type(var_41.known_future_library).__module__}.{type(var_41.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.known_future_library) == 1
    assert f'{type(var_41.known_third_party).__module__}.{type(var_41.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.known_third_party) == 0
    assert f'{type(var_41.known_first_party).__module__}.{type(var_41.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.known_first_party) == 0
    assert f'{type(var_41.known_local_folder).__module__}.{type(var_41.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.known_local_folder) == 0
    assert f'{type(var_41.known_standard_library).__module__}.{type(var_41.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.known_standard_library) == 234
    assert f'{type(var_41.extra_standard_library).__module__}.{type(var_41.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.extra_standard_library) == 0
    assert var_41.known_other == {}
    assert var_41.multi_line_output == module_1.WrapModes.GRID
    assert var_41.forced_separate == ()
    assert var_41.indent == '    '
    assert var_41.comment_prefix == '  #'
    assert var_41.length_sort is False
    assert var_41.length_sort_straight is False
    assert f'{type(var_41.length_sort_sections).__module__}.{type(var_41.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.length_sort_sections) == 0
    assert f'{type(var_41.add_imports).__module__}.{type(var_41.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.add_imports) == 0
    assert f'{type(var_41.remove_imports).__module__}.{type(var_41.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.remove_imports) == 0
    assert var_41.append_only is False
    assert var_41.reverse_relative is False
    assert var_41.force_single_line is False
    assert var_41.single_line_exclusions == ()
    assert var_41.default_section == 'THIRDPARTY'
    assert var_41.import_headings == {}
    assert var_41.import_footers == {}
    assert var_41.balanced_wrapping is False
    assert var_41.use_parentheses is False
    assert var_41.order_by_type is True
    assert var_41.atomic is False
    assert var_41.lines_before_imports == -1
    assert var_41.lines_after_imports == -1
    assert var_41.lines_between_sections == 1
    assert var_41.lines_between_types == 0
    assert var_41.combine_as_imports is False
    assert var_41.combine_star is False
    assert var_41.include_trailing_comma is False
    assert var_41.from_first is False
    assert var_41.verbose is False
    assert var_41.quiet is False
    assert var_41.force_adds is False
    assert var_41.force_alphabetical_sort_within_sections is False
    assert var_41.force_alphabetical_sort is False
    assert var_41.force_grid_wrap == 0
    assert var_41.force_sort_within_sections is False
    assert var_41.lexicographical is False
    assert var_41.group_by_package is False
    assert var_41.ignore_whitespace is False
    assert f'{type(var_41.no_lines_before).__module__}.{type(var_41.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.no_lines_before) == 0
    assert var_41.no_inline_sort is False
    assert var_41.ignore_comments is False
    assert var_41.case_sensitive is False
    assert f'{type(var_41.sources).__module__}.{type(var_41.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_41.sources) == 1
    assert var_41.virtual_env == ''
    assert var_41.conda_env == ''
    assert var_41.ensure_newline_before_comments is False
    assert var_41.directory == '/workspace'
    assert var_41.profile == ''
    assert var_41.honor_noqa is False
    assert f'{type(var_41.src_paths).__module__}.{type(var_41.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_41.src_paths) == 2
    assert var_41.remove_redundant_aliases is False
    assert var_41.float_to_top is False
    assert var_41.filter_files is False
    assert var_41.formatter == ''
    assert var_41.formatting_function is None
    assert var_41.color_output is False
    assert f'{type(var_41.treat_comments_as_code).__module__}.{type(var_41.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.treat_comments_as_code) == 0
    assert var_41.treat_all_comments_as_code is False
    assert f'{type(var_41.supported_extensions).__module__}.{type(var_41.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.supported_extensions) == 4
    assert f'{type(var_41.blocked_extensions).__module__}.{type(var_41.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.blocked_extensions) == 1
    assert f'{type(var_41.constants).__module__}.{type(var_41.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.constants) == 0
    assert f'{type(var_41.classes).__module__}.{type(var_41.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.classes) == 0
    assert f'{type(var_41.variables).__module__}.{type(var_41.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.variables) == 0
    assert var_41.dedup_headings is False
    assert var_41.only_sections is False
    assert var_41.only_modified is False
    assert var_41.combine_straight_imports is False
    assert var_41.auto_identify_namespace_packages is True
    assert f'{type(var_41.namespace_packages).__module__}.{type(var_41.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_41.namespace_packages) == 0
    assert var_41.follow_links is True
    assert var_41.indented_import_headings is True
    assert var_41.honor_case_in_force_sorted_sections is False
    assert var_41.sort_relative_in_force_sorted_sections is False
    assert var_41.overwrite_in_place is False
    assert var_41.reverse_sort is False
    assert var_41.star_first is False
    assert var_41.git_ls_files == {}
    assert var_41.format_error == '{error}: {message}'
    assert var_41.format_success == '{success}: {message}'
    assert var_41.sort_order == 'natural'
    assert var_41.sort_reexports is False
    assert var_41.split_on_trailing_comma is False
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
    var_42 = 'import os\n'
    var_43 = module_0.file_contents(var_42, var_41)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_43) == 14
    var_44 = 'from os import path as p  # comment\n'
    var_45 = module_0.file_contents(var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_45) == 14
    var_46 = var_45.categorized_comments
    var_47 = '# isort: stdlib'
    var_48 = [var_47]
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
    var_50 = '# isort: stdlib\nimport os\n'
    var_51 = module_0.file_contents(var_50, var_49)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_51) == 14

def test_case_22():
    var_0 = 'from libc.stdlib cimport malloc, free\n'
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

def test_case_23():
    var_0 = 'import os  # operating system\nimport sys  # system\n'
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
    var_0 = '# isort:imports-FUTURE\nimport os\n'
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
    var_3 = var_1.import_placements

def test_case_25():
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
    var_12 = 'import numpy as np\nfrom os import path as p\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'straight'
    var_15 = var_13.as_map[var_14]
    var_16 = len(var_15)
    var_17 = 0
    var_18 = var_16 > var_17
    var_19 = 'from os impo>t (\n    path,\n    getcwd\n)\n'
    var_20 = module_0.file_contents(var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_20) == 14
    var_21 = 'from os import path, \\\n    getcwd\n'
    var_22 = module_0.file_contents(var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_22) == 14
    with pytest.raises(AttributeError):
        var_23 = var_3.lines_without_imports

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
    var_12 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = 'import numpy as np\nfrom os import path as p\n'
    var_15 = module_0.file_contents(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_15) == 14
    var_16 = 'straight'
    var_17 = var_15.as_map[var_16]
    var_18 = len(var_17)
    var_19 = 0
    var_20 = var_18 > var_19
    var_21 = 'from'
    var_22 = var_15.as_map[var_21]
    var_23 = len(var_22)
    var_24 = var_23 > var_19
    var_25 = 'import os\n\ndef foo():\n    pass\n'
    var_26 = module_0.file_contents(var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_26) == 14
    var_27 = var_26.lines_without_imports
    var_28 = len(var_27)
    var_29 = ''
    var_30 = module_0.file_contents(var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_30) == 14
    var_31 = '# This is a comment\n# Another comment\n'
    var_32 = module_0.file_contents(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_32) == 14
    var_33 = var_32.lines_without_imports
    var_34 = len(var_33)
    var_35 = 'import os; import sys\n'
    var_36 = module_0.file_contents(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = 'from os import \\\n    path\n'
    var_38 = module_0.file_contents(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_38) == 14
    var_39 = '\r\n'
    var_40 = module_3.Config()
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'isort.settings.Config'
    assert var_40.py_version == 'py3'
    assert f'{type(var_40.force_to_top).__module__}.{type(var_40.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.force_to_top) == 0
    assert f'{type(var_40.skip).__module__}.{type(var_40.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.skip) == 19
    assert f'{type(var_40.extend_skip).__module__}.{type(var_40.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.extend_skip) == 0
    assert f'{type(var_40.skip_glob).__module__}.{type(var_40.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.skip_glob) == 0
    assert f'{type(var_40.extend_skip_glob).__module__}.{type(var_40.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.extend_skip_glob) == 0
    assert var_40.skip_gitignore is False
    assert var_40.line_length == 79
    assert var_40.wrap_length == 0
    assert var_40.line_ending == ''
    assert var_40.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_40.no_sections is False
    assert f'{type(var_40.known_future_library).__module__}.{type(var_40.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.known_future_library) == 1
    assert f'{type(var_40.known_third_party).__module__}.{type(var_40.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.known_third_party) == 0
    assert f'{type(var_40.known_first_party).__module__}.{type(var_40.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.known_first_party) == 0
    assert f'{type(var_40.known_local_folder).__module__}.{type(var_40.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.known_local_folder) == 0
    assert f'{type(var_40.known_standard_library).__module__}.{type(var_40.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.known_standard_library) == 234
    assert f'{type(var_40.extra_standard_library).__module__}.{type(var_40.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.extra_standard_library) == 0
    assert var_40.known_other == {}
    assert var_40.multi_line_output == module_1.WrapModes.GRID
    assert var_40.forced_separate == ()
    assert var_40.indent == '    '
    assert var_40.comment_prefix == '  #'
    assert var_40.length_sort is False
    assert var_40.length_sort_straight is False
    assert f'{type(var_40.length_sort_sections).__module__}.{type(var_40.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.length_sort_sections) == 0
    assert f'{type(var_40.add_imports).__module__}.{type(var_40.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.add_imports) == 0
    assert f'{type(var_40.remove_imports).__module__}.{type(var_40.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.remove_imports) == 0
    assert var_40.append_only is False
    assert var_40.reverse_relative is False
    assert var_40.force_single_line is False
    assert var_40.single_line_exclusions == ()
    assert var_40.default_section == 'THIRDPARTY'
    assert var_40.import_headings == {}
    assert var_40.import_footers == {}
    assert var_40.balanced_wrapping is False
    assert var_40.use_parentheses is False
    assert var_40.order_by_type is True
    assert var_40.atomic is False
    assert var_40.lines_before_imports == -1
    assert var_40.lines_after_imports == -1
    assert var_40.lines_between_sections == 1
    assert var_40.lines_between_types == 0
    assert var_40.combine_as_imports is False
    assert var_40.combine_star is False
    assert var_40.include_trailing_comma is False
    assert var_40.from_first is False
    assert var_40.verbose is False
    assert var_40.quiet is False
    assert var_40.force_adds is False
    assert var_40.force_alphabetical_sort_within_sections is False
    assert var_40.force_alphabetical_sort is False
    assert var_40.force_grid_wrap == 0
    assert var_40.force_sort_within_sections is False
    assert var_40.lexicographical is False
    assert var_40.group_by_package is False
    assert var_40.ignore_whitespace is False
    assert f'{type(var_40.no_lines_before).__module__}.{type(var_40.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.no_lines_before) == 0
    assert var_40.no_inline_sort is False
    assert var_40.ignore_comments is False
    assert var_40.case_sensitive is False
    assert f'{type(var_40.sources).__module__}.{type(var_40.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_40.sources) == 1
    assert var_40.virtual_env == ''
    assert var_40.conda_env == ''
    assert var_40.ensure_newline_before_comments is False
    assert var_40.directory == '/workspace'
    assert var_40.profile == ''
    assert var_40.honor_noqa is False
    assert f'{type(var_40.src_paths).__module__}.{type(var_40.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_40.src_paths) == 2
    assert var_40.remove_redundant_aliases is False
    assert var_40.float_to_top is False
    assert var_40.filter_files is False
    assert var_40.formatter == ''
    assert var_40.formatting_function is None
    assert var_40.color_output is False
    assert f'{type(var_40.treat_comments_as_code).__module__}.{type(var_40.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.treat_comments_as_code) == 0
    assert var_40.treat_all_comments_as_code is False
    assert f'{type(var_40.supported_extensions).__module__}.{type(var_40.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.supported_extensions) == 4
    assert f'{type(var_40.blocked_extensions).__module__}.{type(var_40.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.blocked_extensions) == 1
    assert f'{type(var_40.constants).__module__}.{type(var_40.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.constants) == 0
    assert f'{type(var_40.classes).__module__}.{type(var_40.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.classes) == 0
    assert f'{type(var_40.variables).__module__}.{type(var_40.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.variables) == 0
    assert var_40.dedup_headings is False
    assert var_40.only_sections is False
    assert var_40.only_modified is False
    assert var_40.combine_straight_imports is False
    assert var_40.auto_identify_namespace_packages is True
    assert f'{type(var_40.namespace_packages).__module__}.{type(var_40.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_40.namespace_packages) == 0
    assert var_40.follow_links is True
    assert var_40.indented_import_headings is True
    assert var_40.honor_case_in_force_sorted_sections is False
    assert var_40.sort_relative_in_force_sorted_sections is False
    assert var_40.overwrite_in_place is False
    assert var_40.reverse_sort is False
    assert var_40.star_first is False
    assert var_40.git_ls_files == {}
    assert var_40.format_error == '{error}: {message}'
    assert var_40.format_success == '{success}: {message}'
    assert var_40.sort_order == 'natural'
    assert var_40.sort_reexports is False
    assert var_40.split_on_trailing_comma is False
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
    var_41 = 'import os\r\nimport sys\r\n'
    var_42 = module_0.file_contents(var_41, var_40)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_42) == 14
    var_43 = 'import os  # isort:skip\nimport sys\n'
    var_44 = module_0.file_contents(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_44) == 14
    var_45 = 'from os import (\n    path,\n)\n'
    var_46 = module_0.file_contents(var_45)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_46) == 14
    var_47 = var_46.trailing_commas
    var_48 = len(var_47)
    var_49 = var_48 >= var_19
    var_50 = 'from os import path  # path comment\nfrom os import getcwd  # getcwd comment\n'
    var_51 = module_0.file_contents(var_50)
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_51) == 14
    var_52 = 'import os\nimport sys\n'
    var_53 = module_0.file_contents(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_53) == 14
    var_54 = var_53.lines_without_imports
    var_55 = len(var_54)
    var_56 = var_53.original_line_count
    var_57 = var_55 - var_56
    var_58 = '# isort:imports-STDLIB\nimport os\n'
    var_59 = module_0.file_contents(var_58)
    assert f'{type(var_59).__module__}.{type(var_59).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_59) == 14
    var_60 = var_59.place_imports
    var_61 = len(var_60)
    var_62 = 'x = "import os"\n'
    var_63 = module_0.file_contents(var_62)
    assert f'{type(var_63).__module__}.{type(var_63).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_63) == 14
    var_64 = '"""\nimport os\n"""\nimport sys\n'
    var_65 = module_0.file_contents(var_64)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_65) == 14
    var_66 = 'from os import path\nfrom os import getcwd\n'
    var_67 = module_0.file_contents(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_67) == 14
    var_68 = 'from libc.stdlib cimport malloc\n'
    var_69 = module_0.file_contents(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_69) == 14

def test_case_27():
    var_0 = 'from os import (\n    path,  # the path\n    getcwd  # get current\n)\n'
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

def test_case_28():
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
    var_8 = 'import os\nfrom sys import argv\n'
    var_9 = module_0.file_contents(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_9) == 14
    var_10 = var_9.imports
    var_11 = len(var_10)
    var_12 = 'import os  # operating system\nfrom sys import argv  # arguments\n'
    var_13 = module_0.file_contents(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_13) == 14
    var_14 = var_13.categorized_comments
    var_15 = len(var_14)
    var_16 = 'from os import (\n    path,\n    getcwd\n)\n'
    var_17 = module_0.file_contents(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_17) == 14
    var_18 = var_17.imports
    var_19 = len(var_18)
    var_20 = 'import numpy as np\nfrom os import path as p\n'
    var_21 = module_0.file_contents(var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_21) == 14
    var_22 = 'straight'
    var_23 = var_21.as_map[var_22]
    var_24 = len(var_23)
    var_25 = 0
    var_26 = var_24 > var_25
    var_27 = 'from'
    var_28 = var_21.as_map[var_27]
    var_29 = len(var_28)
    var_30 = var_29 > var_25
    var_31 = 'import os\n\ndef foo():\n    pass\n'
    var_32 = module_0.file_contents(var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_32) == 14
    var_33 = var_32.lines_without_imports
    var_34 = len(var_33)
    var_35 = ''
    var_36 = module_0.file_contents(var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_36) == 14
    var_37 = '# This is a comment\n# Another comment\n'
    var_38 = module_0.file_contents(var_37)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_38) == 14
    var_39 = 'import os, \\\n    sys\n'
    var_40 = module_0.file_contents(var_39)
    assert f'{type(var_40).__module__}.{type(var_40).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_40) == 14
    var_41 = 'import os; import sys\n'
    var_42 = module_0.file_contents(var_41)
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_42) == 14
    var_43 = 'from os import path,\n'
    var_44 = module_0.file_contents(var_43)
    assert f'{type(var_44).__module__}.{type(var_44).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_44) == 14
    var_45 = var_44.trailing_commas
    var_46 = len(var_45)
    var_47 = 'import os\nimport sys\n'
    var_48 = module_0.file_contents(var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_48) == 14
    var_49 = 'import os\nimport sys\n'
    var_50 = module_0.file_contents(var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_50) == 14
    var_51 = var_50.change_count
    var_52 = 'import os\n'
    var_53 = module_0.file_contents(var_52)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_53) == 14
    var_54 = var_53.sections
    var_55 = len(var_54)
    var_56 = True
    var_57 = False
    var_58 = module_3.Config()
    assert f'{type(var_58).__module__}.{type(var_58).__qualname__}' == 'isort.settings.Config'
    assert var_58.py_version == 'py3'
    assert f'{type(var_58.force_to_top).__module__}.{type(var_58.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.force_to_top) == 0
    assert f'{type(var_58.skip).__module__}.{type(var_58.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.skip) == 19
    assert f'{type(var_58.extend_skip).__module__}.{type(var_58.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.extend_skip) == 0
    assert f'{type(var_58.skip_glob).__module__}.{type(var_58.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.skip_glob) == 0
    assert f'{type(var_58.extend_skip_glob).__module__}.{type(var_58.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.extend_skip_glob) == 0
    assert var_58.skip_gitignore is False
    assert var_58.line_length == 79
    assert var_58.wrap_length == 0
    assert var_58.line_ending == ''
    assert var_58.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_58.no_sections is False
    assert f'{type(var_58.known_future_library).__module__}.{type(var_58.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.known_future_library) == 1
    assert f'{type(var_58.known_third_party).__module__}.{type(var_58.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.known_third_party) == 0
    assert f'{type(var_58.known_first_party).__module__}.{type(var_58.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.known_first_party) == 0
    assert f'{type(var_58.known_local_folder).__module__}.{type(var_58.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.known_local_folder) == 0
    assert f'{type(var_58.known_standard_library).__module__}.{type(var_58.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.known_standard_library) == 234
    assert f'{type(var_58.extra_standard_library).__module__}.{type(var_58.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.extra_standard_library) == 0
    assert var_58.known_other == {}
    assert var_58.multi_line_output == module_1.WrapModes.GRID
    assert var_58.forced_separate == ()
    assert var_58.indent == '    '
    assert var_58.comment_prefix == '  #'
    assert var_58.length_sort is False
    assert var_58.length_sort_straight is False
    assert f'{type(var_58.length_sort_sections).__module__}.{type(var_58.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.length_sort_sections) == 0
    assert f'{type(var_58.add_imports).__module__}.{type(var_58.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.add_imports) == 0
    assert f'{type(var_58.remove_imports).__module__}.{type(var_58.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.remove_imports) == 0
    assert var_58.append_only is False
    assert var_58.reverse_relative is False
    assert var_58.force_single_line is False
    assert var_58.single_line_exclusions == ()
    assert var_58.default_section == 'THIRDPARTY'
    assert var_58.import_headings == {}
    assert var_58.import_footers == {}
    assert var_58.balanced_wrapping is False
    assert var_58.use_parentheses is False
    assert var_58.order_by_type is True
    assert var_58.atomic is False
    assert var_58.lines_before_imports == -1
    assert var_58.lines_after_imports == -1
    assert var_58.lines_between_sections == 1
    assert var_58.lines_between_types == 0
    assert var_58.combine_as_imports is False
    assert var_58.combine_star is False
    assert var_58.include_trailing_comma is False
    assert var_58.from_first is False
    assert var_58.verbose is False
    assert var_58.quiet is False
    assert var_58.force_adds is False
    assert var_58.force_alphabetical_sort_within_sections is False
    assert var_58.force_alphabetical_sort is False
    assert var_58.force_grid_wrap == 0
    assert var_58.force_sort_within_sections is False
    assert var_58.lexicographical is False
    assert var_58.group_by_package is False
    assert var_58.ignore_whitespace is False
    assert f'{type(var_58.no_lines_before).__module__}.{type(var_58.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.no_lines_before) == 0
    assert var_58.no_inline_sort is False
    assert var_58.ignore_comments is False
    assert var_58.case_sensitive is False
    assert f'{type(var_58.sources).__module__}.{type(var_58.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_58.sources) == 1
    assert var_58.virtual_env == ''
    assert var_58.conda_env == ''
    assert var_58.ensure_newline_before_comments is False
    assert var_58.directory == '/workspace'
    assert var_58.profile == ''
    assert var_58.honor_noqa is False
    assert f'{type(var_58.src_paths).__module__}.{type(var_58.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_58.src_paths) == 2
    assert var_58.remove_redundant_aliases is False
    assert var_58.float_to_top is False
    assert var_58.filter_files is False
    assert var_58.formatter == ''
    assert var_58.formatting_function is None
    assert var_58.color_output is False
    assert f'{type(var_58.treat_comments_as_code).__module__}.{type(var_58.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.treat_comments_as_code) == 0
    assert var_58.treat_all_comments_as_code is False
    assert f'{type(var_58.supported_extensions).__module__}.{type(var_58.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.supported_extensions) == 4
    assert f'{type(var_58.blocked_extensions).__module__}.{type(var_58.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.blocked_extensions) == 1
    assert f'{type(var_58.constants).__module__}.{type(var_58.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.constants) == 0
    assert f'{type(var_58.classes).__module__}.{type(var_58.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.classes) == 0
    assert f'{type(var_58.variables).__module__}.{type(var_58.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.variables) == 0
    assert var_58.dedup_headings is False
    assert var_58.only_sections is False
    assert var_58.only_modified is False
    assert var_58.combine_straight_imports is False
    assert var_58.auto_identify_namespace_packages is True
    assert f'{type(var_58.namespace_packages).__module__}.{type(var_58.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_58.namespace_packages) == 0
    assert var_58.follow_links is True
    assert var_58.indented_import_headings is True
    assert var_58.honor_case_in_force_sorted_sections is False
    assert var_58.sort_relative_in_force_sorted_sections is False
    assert var_58.overwrite_in_place is False
    assert var_58.reverse_sort is False
    assert var_58.star_first is False
    assert var_58.git_ls_files == {}
    assert var_58.format_error == '{error}: {message}'
    assert var_58.format_success == '{success}: {message}'
    assert var_58.sort_order == 'natural'
    assert var_58.sort_reexports is False
    assert var_58.split_on_trailing_comma is False
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
    var_59 = 'import os\n'
    var_60 = module_0.file_contents(var_59, var_58)
    assert f'{type(var_60).__module__}.{type(var_60).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_60) == 14
    var_61 = var_60.verbose_output
    var_62 = '\r\n'
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
    var_64 = 'import os\r\nimport sys\r\n'
    var_65 = module_0.file_contents(var_64, var_63)
    assert f'{type(var_65).__module__}.{type(var_65).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_65) == 14
    var_66 = 'from os import (\n    path,  # path module\n    getcwd  # get current directory\n)\n'
    var_67 = module_0.file_contents(var_66)
    assert f'{type(var_67).__module__}.{type(var_67).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_67) == 14
    var_68 = 'import os\nimport sys\n'
    var_69 = module_0.file_contents(var_68)
    assert f'{type(var_69).__module__}.{type(var_69).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_69) == 14
    var_70 = var_69.in_lines
    var_71 = len(var_70)
    var_72 = '# isort:imports-THIRDPARTY\nimport custom_module\n'
    var_73 = module_0.file_contents(var_72)
    assert f'{type(var_73).__module__}.{type(var_73).__qualname__}' == 'isort.parse.ParsedContent'
    assert len(var_73) == 14
    var_74 = var_73.place_imports
    var_75 = len(var_74)