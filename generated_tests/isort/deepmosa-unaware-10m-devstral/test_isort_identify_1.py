# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.identify as module_0
import isort.wrap_modes as module_1
import isort.settings as module_2

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.Import(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.identify.Import'
    assert len(var_2) == 7
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_3 = var_2.statement()
    assert var_3 == 'import None'

def test_case_1():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.Import(*var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.identify.Import'
    assert len(var_2) == 7
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_3 = var_2.__str__()
    assert var_3 == ':None import None'

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.Import()

def test_case_3():
    var_0 = 'from pathlib import Path as P\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)

def test_case_4():
    var_0 = 'import nmpy as n\n'
    var_1 = '>'
    var_2 = True
    var_3 = 'import sys, os\n'
    var_4 = [var_0]
    var_5 = module_0.imports(var_4)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_6 = module_2.Config()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.settings.Config'
    assert var_6.py_version == 'py3'
    assert f'{type(var_6.force_to_top).__module__}.{type(var_6.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.force_to_top) == 0
    assert f'{type(var_6.skip).__module__}.{type(var_6.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.skip) == 19
    assert f'{type(var_6.extend_skip).__module__}.{type(var_6.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.extend_skip) == 0
    assert f'{type(var_6.skip_glob).__module__}.{type(var_6.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.skip_glob) == 0
    assert f'{type(var_6.extend_skip_glob).__module__}.{type(var_6.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.extend_skip_glob) == 0
    assert var_6.skip_gitignore is False
    assert var_6.line_length == 79
    assert var_6.wrap_length == 0
    assert var_6.line_ending == ''
    assert var_6.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_6.no_sections is False
    assert f'{type(var_6.known_future_library).__module__}.{type(var_6.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_future_library) == 1
    assert f'{type(var_6.known_third_party).__module__}.{type(var_6.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_third_party) == 0
    assert f'{type(var_6.known_first_party).__module__}.{type(var_6.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_first_party) == 0
    assert f'{type(var_6.known_local_folder).__module__}.{type(var_6.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_local_folder) == 0
    assert f'{type(var_6.known_standard_library).__module__}.{type(var_6.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.known_standard_library) == 234
    assert f'{type(var_6.extra_standard_library).__module__}.{type(var_6.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.extra_standard_library) == 0
    assert var_6.known_other == {}
    assert var_6.multi_line_output == module_1.WrapModes.GRID
    assert var_6.forced_separate == ()
    assert var_6.indent == '    '
    assert var_6.comment_prefix == '  #'
    assert var_6.length_sort is False
    assert var_6.length_sort_straight is False
    assert f'{type(var_6.length_sort_sections).__module__}.{type(var_6.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.length_sort_sections) == 0
    assert f'{type(var_6.add_imports).__module__}.{type(var_6.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.add_imports) == 0
    assert f'{type(var_6.remove_imports).__module__}.{type(var_6.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.remove_imports) == 0
    assert var_6.append_only is False
    assert var_6.reverse_relative is False
    assert var_6.force_single_line is False
    assert var_6.single_line_exclusions == ()
    assert var_6.default_section == 'THIRDPARTY'
    assert var_6.import_headings == {}
    assert var_6.import_footers == {}
    assert var_6.balanced_wrapping is False
    assert var_6.use_parentheses is False
    assert var_6.order_by_type is True
    assert var_6.atomic is False
    assert var_6.lines_before_imports == -1
    assert var_6.lines_after_imports == -1
    assert var_6.lines_between_sections == 1
    assert var_6.lines_between_types == 0
    assert var_6.combine_as_imports is False
    assert var_6.combine_star is False
    assert var_6.include_trailing_comma is False
    assert var_6.from_first is False
    assert var_6.verbose is False
    assert var_6.quiet is False
    assert var_6.force_adds is False
    assert var_6.force_alphabetical_sort_within_sections is False
    assert var_6.force_alphabetical_sort is False
    assert var_6.force_grid_wrap == 0
    assert var_6.force_sort_within_sections is False
    assert var_6.lexicographical is False
    assert var_6.group_by_package is False
    assert var_6.ignore_whitespace is False
    assert f'{type(var_6.no_lines_before).__module__}.{type(var_6.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.no_lines_before) == 0
    assert var_6.no_inline_sort is False
    assert var_6.ignore_comments is False
    assert var_6.case_sensitive is False
    assert f'{type(var_6.sources).__module__}.{type(var_6.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_6.sources) == 1
    assert var_6.virtual_env == ''
    assert var_6.conda_env == ''
    assert var_6.ensure_newline_before_comments is False
    assert var_6.directory == '/workspace'
    assert var_6.profile == ''
    assert var_6.honor_noqa is False
    assert f'{type(var_6.src_paths).__module__}.{type(var_6.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_6.src_paths) == 2
    assert var_6.remove_redundant_aliases is False
    assert var_6.float_to_top is False
    assert var_6.filter_files is False
    assert var_6.formatter == ''
    assert var_6.formatting_function is None
    assert var_6.color_output is False
    assert f'{type(var_6.treat_comments_as_code).__module__}.{type(var_6.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.treat_comments_as_code) == 0
    assert var_6.treat_all_comments_as_code is False
    assert f'{type(var_6.supported_extensions).__module__}.{type(var_6.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.supported_extensions) == 4
    assert f'{type(var_6.blocked_extensions).__module__}.{type(var_6.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.blocked_extensions) == 1
    assert f'{type(var_6.constants).__module__}.{type(var_6.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.constants) == 0
    assert f'{type(var_6.classes).__module__}.{type(var_6.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.classes) == 0
    assert f'{type(var_6.variables).__module__}.{type(var_6.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.variables) == 0
    assert var_6.dedup_headings is False
    assert var_6.only_sections is False
    assert var_6.only_modified is False
    assert var_6.combine_straight_imports is False
    assert var_6.auto_identify_namespace_packages is True
    assert f'{type(var_6.namespace_packages).__module__}.{type(var_6.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_6.namespace_packages) == 0
    assert var_6.follow_links is True
    assert var_6.indented_import_headings is True
    assert var_6.honor_case_in_force_sorted_sections is False
    assert var_6.sort_relative_in_force_sorted_sections is False
    assert var_6.overwrite_in_place is False
    assert var_6.reverse_sort is False
    assert var_6.star_first is False
    assert var_6.git_ls_files == {}
    assert var_6.format_error == '{error}: {message}'
    assert var_6.format_success == '{success}: {message}'
    assert var_6.sort_order == 'natural'
    assert var_6.sort_reexports is False
    assert var_6.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_7 = 'import numpy as numpy\n'
    var_8 = [var_7]
    var_9 = module_0.imports(var_8, var_6)
    var_10 = list(var_9)
    var_11 = 'def foo():\n'
    var_12 = [var_1, var_3, var_11, var_0, var_0]
    var_13 = module_0.imports(var_12, top_only=var_10)
    var_14 = list(var_13)

def test_case_5():
    var_0 = 'impor o\n'
    var_1 = '\x0b4u2Q<dh:\\np{dm'
    var_2 = 'from pandas import DaaFrame as DF\n'
    var_3 = [var_2]
    var_4 = module_0.imports(var_3)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_5 = [var_2]
    var_6 = module_0.imports(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = var_7.__eq__(var_7)
    assert var_9 is True
    var_10 = [var_1]
    var_11 = module_0.imports(var_10)
    var_12 = 'from typing import (\n    List,\n    Dict,\n)$'
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = 'import os  # Some comment\n'
    var_16 = True
    var_17 = module_2.Config()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'isort.settings.Config'
    assert var_17.py_version == 'py3'
    assert f'{type(var_17.force_to_top).__module__}.{type(var_17.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.force_to_top) == 0
    assert f'{type(var_17.skip).__module__}.{type(var_17.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.skip) == 19
    assert f'{type(var_17.extend_skip).__module__}.{type(var_17.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extend_skip) == 0
    assert f'{type(var_17.skip_glob).__module__}.{type(var_17.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.skip_glob) == 0
    assert f'{type(var_17.extend_skip_glob).__module__}.{type(var_17.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extend_skip_glob) == 0
    assert var_17.skip_gitignore is False
    assert var_17.line_length == 79
    assert var_17.wrap_length == 0
    assert var_17.line_ending == ''
    assert var_17.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_17.no_sections is False
    assert f'{type(var_17.known_future_library).__module__}.{type(var_17.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_future_library) == 1
    assert f'{type(var_17.known_third_party).__module__}.{type(var_17.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_third_party) == 0
    assert f'{type(var_17.known_first_party).__module__}.{type(var_17.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_first_party) == 0
    assert f'{type(var_17.known_local_folder).__module__}.{type(var_17.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_local_folder) == 0
    assert f'{type(var_17.known_standard_library).__module__}.{type(var_17.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_standard_library) == 234
    assert f'{type(var_17.extra_standard_library).__module__}.{type(var_17.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extra_standard_library) == 0
    assert var_17.known_other == {}
    assert var_17.multi_line_output == module_1.WrapModes.GRID
    assert var_17.forced_separate == ()
    assert var_17.indent == '    '
    assert var_17.comment_prefix == '  #'
    assert var_17.length_sort is False
    assert var_17.length_sort_straight is False
    assert f'{type(var_17.length_sort_sections).__module__}.{type(var_17.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.length_sort_sections) == 0
    assert f'{type(var_17.add_imports).__module__}.{type(var_17.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.add_imports) == 0
    assert f'{type(var_17.remove_imports).__module__}.{type(var_17.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.remove_imports) == 0
    assert var_17.append_only is False
    assert var_17.reverse_relative is False
    assert var_17.force_single_line is False
    assert var_17.single_line_exclusions == ()
    assert var_17.default_section == 'THIRDPARTY'
    assert var_17.import_headings == {}
    assert var_17.import_footers == {}
    assert var_17.balanced_wrapping is False
    assert var_17.use_parentheses is False
    assert var_17.order_by_type is True
    assert var_17.atomic is False
    assert var_17.lines_before_imports == -1
    assert var_17.lines_after_imports == -1
    assert var_17.lines_between_sections == 1
    assert var_17.lines_between_types == 0
    assert var_17.combine_as_imports is False
    assert var_17.combine_star is False
    assert var_17.include_trailing_comma is False
    assert var_17.from_first is False
    assert var_17.verbose is False
    assert var_17.quiet is False
    assert var_17.force_adds is False
    assert var_17.force_alphabetical_sort_within_sections is False
    assert var_17.force_alphabetical_sort is False
    assert var_17.force_grid_wrap == 0
    assert var_17.force_sort_within_sections is False
    assert var_17.lexicographical is False
    assert var_17.group_by_package is False
    assert var_17.ignore_whitespace is False
    assert f'{type(var_17.no_lines_before).__module__}.{type(var_17.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.no_lines_before) == 0
    assert var_17.no_inline_sort is False
    assert var_17.ignore_comments is False
    assert var_17.case_sensitive is False
    assert f'{type(var_17.sources).__module__}.{type(var_17.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_17.sources) == 1
    assert var_17.virtual_env == ''
    assert var_17.conda_env == ''
    assert var_17.ensure_newline_before_comments is False
    assert var_17.directory == '/workspace'
    assert var_17.profile == ''
    assert var_17.honor_noqa is False
    assert f'{type(var_17.src_paths).__module__}.{type(var_17.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_17.src_paths) == 2
    assert var_17.remove_redundant_aliases is False
    assert var_17.float_to_top is False
    assert var_17.filter_files is False
    assert var_17.formatter == ''
    assert var_17.formatting_function is None
    assert var_17.color_output is False
    assert f'{type(var_17.treat_comments_as_code).__module__}.{type(var_17.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.treat_comments_as_code) == 0
    assert var_17.treat_all_comments_as_code is False
    assert f'{type(var_17.supported_extensions).__module__}.{type(var_17.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.supported_extensions) == 4
    assert f'{type(var_17.blocked_extensions).__module__}.{type(var_17.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.blocked_extensions) == 1
    assert f'{type(var_17.constants).__module__}.{type(var_17.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.constants) == 0
    assert f'{type(var_17.classes).__module__}.{type(var_17.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.classes) == 0
    assert f'{type(var_17.variables).__module__}.{type(var_17.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.variables) == 0
    assert var_17.dedup_headings is False
    assert var_17.only_sections is False
    assert var_17.only_modified is False
    assert var_17.combine_straight_imports is False
    assert var_17.auto_identify_namespace_packages is True
    assert f'{type(var_17.namespace_packages).__module__}.{type(var_17.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.namespace_packages) == 0
    assert var_17.follow_links is True
    assert var_17.indented_import_headings is True
    assert var_17.honor_case_in_force_sorted_sections is False
    assert var_17.sort_relative_in_force_sorted_sections is False
    assert var_17.overwrite_in_place is False
    assert var_17.reverse_sort is False
    assert var_17.star_first is False
    assert var_17.git_ls_files == {}
    assert var_17.format_error == '{error}: {message}'
    assert var_17.format_success == '{success}: {message}'
    assert var_17.sort_order == 'natural'
    assert var_17.sort_reexports is False
    assert var_17.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_18 = 'import numpy as numpy\n'
    var_19 = [var_18]
    var_20 = module_0.imports(var_19, var_17)
    var_21 = list(var_20)
    var_22 = '"""Module docstring"""'
    var_23 = [var_22, var_0]
    var_24 = module_0.imports(var_23)
    var_25 = '0-'
    var_26 = [var_18, var_25]
    var_27 = module_0.imports(var_26)
    var_28 = '    yield\n'
    var_29 = [var_15, var_28]
    var_30 = module_0.imports(var_29)
    var_31 = list(var_30)
    with pytest.raises(TypeError):
        var_32 = len(var_8)
    assert var_32 == 1

def test_case_6():
    var_0 = 'import numpy as np\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_7():
    var_0 = 'from typing import (\n'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = False
    with pytest.raises(TypeError):
        var_5 = list(var_4)

def test_case_8():
    var_0 = 'fro} collections import defaultdict\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = len(var_3)
    var_5 = 'from pathlib import Path as P\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1

def test_case_9():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = 'import numpy as np\n'
    var_5 = [var_4]
    var_6 = module_0.imports(var_5)
    var_7 = list(var_6)
    var_8 = [var_4]
    var_9 = module_0.imports(var_8)
    var_10 = list(var_9)
    var_11 = var_7.__repr__()
    assert var_11 == "[Import(line_number=1, indented=False, module='numpy', attribute=None, alias='np', cimport=False, file_path=None)]"
    var_12 = list(var_11)
    var_13 = module_0.imports(var_11)
    var_14 = list(var_13)
    var_15 = 'import sys, os\n'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = 'from typing import (\n'
    var_20 = ' t   Dct,\n'
    var_21 = ')\n'
    var_22 = [var_19, var_4, var_20, var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = module_0.imports(var_11)
    var_26 = list(var_25)
    var_27 = 'impot os  # Operating system interfaces'
    var_28 = [var_27]
    var_29 = module_0.imports(var_28)
    var_30 = list(var_29)
    var_31 = True
    var_32 = module_2.Config()
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'isort.settings.Config'
    assert var_32.py_version == 'py3'
    assert f'{type(var_32.force_to_top).__module__}.{type(var_32.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.force_to_top) == 0
    assert f'{type(var_32.skip).__module__}.{type(var_32.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.skip) == 19
    assert f'{type(var_32.extend_skip).__module__}.{type(var_32.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extend_skip) == 0
    assert f'{type(var_32.skip_glob).__module__}.{type(var_32.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.skip_glob) == 0
    assert f'{type(var_32.extend_skip_glob).__module__}.{type(var_32.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extend_skip_glob) == 0
    assert var_32.skip_gitignore is False
    assert var_32.line_length == 79
    assert var_32.wrap_length == 0
    assert var_32.line_ending == ''
    assert var_32.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_32.no_sections is False
    assert f'{type(var_32.known_future_library).__module__}.{type(var_32.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_future_library) == 1
    assert f'{type(var_32.known_third_party).__module__}.{type(var_32.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_third_party) == 0
    assert f'{type(var_32.known_first_party).__module__}.{type(var_32.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_first_party) == 0
    assert f'{type(var_32.known_local_folder).__module__}.{type(var_32.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_local_folder) == 0
    assert f'{type(var_32.known_standard_library).__module__}.{type(var_32.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.known_standard_library) == 234
    assert f'{type(var_32.extra_standard_library).__module__}.{type(var_32.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.extra_standard_library) == 0
    assert var_32.known_other == {}
    assert var_32.multi_line_output == module_1.WrapModes.GRID
    assert var_32.forced_separate == ()
    assert var_32.indent == '    '
    assert var_32.comment_prefix == '  #'
    assert var_32.length_sort is False
    assert var_32.length_sort_straight is False
    assert f'{type(var_32.length_sort_sections).__module__}.{type(var_32.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.length_sort_sections) == 0
    assert f'{type(var_32.add_imports).__module__}.{type(var_32.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.add_imports) == 0
    assert f'{type(var_32.remove_imports).__module__}.{type(var_32.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.remove_imports) == 0
    assert var_32.append_only is False
    assert var_32.reverse_relative is False
    assert var_32.force_single_line is False
    assert var_32.single_line_exclusions == ()
    assert var_32.default_section == 'THIRDPARTY'
    assert var_32.import_headings == {}
    assert var_32.import_footers == {}
    assert var_32.balanced_wrapping is False
    assert var_32.use_parentheses is False
    assert var_32.order_by_type is True
    assert var_32.atomic is False
    assert var_32.lines_before_imports == -1
    assert var_32.lines_after_imports == -1
    assert var_32.lines_between_sections == 1
    assert var_32.lines_between_types == 0
    assert var_32.combine_as_imports is False
    assert var_32.combine_star is False
    assert var_32.include_trailing_comma is False
    assert var_32.from_first is False
    assert var_32.verbose is False
    assert var_32.quiet is False
    assert var_32.force_adds is False
    assert var_32.force_alphabetical_sort_within_sections is False
    assert var_32.force_alphabetical_sort is False
    assert var_32.force_grid_wrap == 0
    assert var_32.force_sort_within_sections is False
    assert var_32.lexicographical is False
    assert var_32.group_by_package is False
    assert var_32.ignore_whitespace is False
    assert f'{type(var_32.no_lines_before).__module__}.{type(var_32.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.no_lines_before) == 0
    assert var_32.no_inline_sort is False
    assert var_32.ignore_comments is False
    assert var_32.case_sensitive is False
    assert f'{type(var_32.sources).__module__}.{type(var_32.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_32.sources) == 1
    assert var_32.virtual_env == ''
    assert var_32.conda_env == ''
    assert var_32.ensure_newline_before_comments is False
    assert var_32.directory == '/workspace'
    assert var_32.profile == ''
    assert var_32.honor_noqa is False
    assert f'{type(var_32.src_paths).__module__}.{type(var_32.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_32.src_paths) == 2
    assert var_32.remove_redundant_aliases is False
    assert var_32.float_to_top is False
    assert var_32.filter_files is False
    assert var_32.formatter == ''
    assert var_32.formatting_function is None
    assert var_32.color_output is False
    assert f'{type(var_32.treat_comments_as_code).__module__}.{type(var_32.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.treat_comments_as_code) == 0
    assert var_32.treat_all_comments_as_code is False
    assert f'{type(var_32.supported_extensions).__module__}.{type(var_32.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.supported_extensions) == 4
    assert f'{type(var_32.blocked_extensions).__module__}.{type(var_32.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.blocked_extensions) == 1
    assert f'{type(var_32.constants).__module__}.{type(var_32.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.constants) == 0
    assert f'{type(var_32.classes).__module__}.{type(var_32.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.classes) == 0
    assert f'{type(var_32.variables).__module__}.{type(var_32.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.variables) == 0
    assert var_32.dedup_headings is False
    assert var_32.only_sections is False
    assert var_32.only_modified is False
    assert var_32.combine_straight_imports is False
    assert var_32.auto_identify_namespace_packages is True
    assert f'{type(var_32.namespace_packages).__module__}.{type(var_32.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_32.namespace_packages) == 0
    assert var_32.follow_links is True
    assert var_32.indented_import_headings is True
    assert var_32.honor_case_in_force_sorted_sections is False
    assert var_32.sort_relative_in_force_sorted_sections is False
    assert var_32.overwrite_in_place is False
    assert var_32.reverse_sort is False
    assert var_32.star_first is False
    assert var_32.git_ls_files == {}
    assert var_32.format_error == '{error}: {message}'
    assert var_32.format_success == '{success}: {message}'
    assert var_32.sort_order == 'natural'
    assert var_32.sort_reexports is False
    assert var_32.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_33 = 'import numpy as numpy\n'
    var_34 = [var_33]
    var_35 = module_0.imports(var_34, var_32)
    var_36 = list(var_35)
    var_37 = True
    var_38 = module_0.imports(var_3, top_only=var_37)
    with pytest.raises(AttributeError):
        var_39 = list(var_38)

def test_case_10():
    var_0 = 'from pathlib mport Path as P\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = len(var_3)

def test_case_11():
    var_0 = 'import os\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = 'import numpy as np\n'
    var_5 = [var_4]
    var_6 = module_0.imports(var_5)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 1
    var_9 = 'from collections import defaultdict\n'
    var_10 = [var_9]
    var_11 = module_0.imports(var_10)
    var_12 = len(var_1)
    assert var_12 == 1
    var_13 = [var_8]
    var_14 = module_0.imports(var_13)
    with pytest.raises(TypeError):
        var_15 = list(var_14)

def test_case_12():
    var_0 = 'from pandas import DaaFrame as DF\n'
    var_1 = [var_0, var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = True
    var_4 = module_2.Config()
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
    assert var_4.multi_line_output == module_1.WrapModes.GRID
    assert var_4.forced_separate == ()
    assert var_4.indent == '    '
    assert var_4.comment_prefix == '  #'
    assert var_4.length_sort is False
    assert var_4.length_sort_straight is False
    assert f'{type(var_4.length_sort_sections).__module__}.{type(var_4.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.length_sort_sections) == 0
    assert f'{type(var_4.add_imports).__module__}.{type(var_4.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_4.add_imports) == 0
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
    assert len(var_4.sources) == 1
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_5 = 'import numpy as numpy\n'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6, var_4)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1

def test_case_13():
    var_0 = 'import numpy as np\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = len(var_0)
    var_5 = 'from pandas import DaaFrame as DF\n'
    var_6 = 'cimport numpy\n'
    var_7 = [var_6]
    var_8 = list(var_2)
    var_9 = 'K;\r3m%xO#j\x0cN'
    var_10 = [var_9, var_5]
    var_11 = module_0.imports(var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    var_14 = 'from typing import (\n    List,\n    Dict,\n)$'
    var_15 = [var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = '{etl'
    var_19 = [var_18]
    var_20 = len(var_9)
    var_21 = module_0.imports(var_19)
    var_22 = list(var_2)
    var_23 = module_0.imports(var_7)
    var_24 = var_22.__repr__()
    assert var_24 == '[]'
    var_25 = list(var_24)
    var_26 = '    yield\n'
    var_27 = [var_24, var_26, var_18]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)

def test_case_14():
    var_0 = "/KTljR<';3T=8"
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = 'import numpy as np\n'
    var_5 = [var_4]
    var_6 = 'np'
    var_7 = module_0.imports(var_5)
    var_8 = list(var_7)
    var_9 = [var_6]
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = var_8.__repr__()
    assert var_12 == "[Import(line_number=1, indented=False, module='numpy', attribute=None, alias='np', cimport=False, file_path=None)]"
    var_13 = list(var_12)
    var_14 = 'cimport numpy\n'
    var_15 = [var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = 'import sys, os\n'
    var_19 = [var_18]
    var_20 = module_0.imports(var_19)
    var_21 = list(var_20)
    var_22 = ' t   Dct,\n'
    var_23 = [var_14, var_4, var_22, var_14]
    var_24 = module_0.imports(var_23)
    var_25 = list(var_24)
    var_26 = '    import sys\n'
    var_27 = module_0.imports(var_12)
    var_28 = list(var_27)
    var_29 = 'impot os  # Operating system interfaces'
    var_30 = [var_29]
    var_31 = module_0.imports(var_30)
    var_32 = list(var_31)
    var_33 = True
    var_34 = module_2.Config()
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_35 = 'import numpy as numpy\n'
    var_36 = [var_35]
    var_37 = module_0.imports(var_36, var_34)
    var_38 = list(var_37)
    var_39 = 'def foo():\n'
    var_40 = [var_0, var_39, var_26]
    var_41 = True
    var_42 = module_0.imports(var_40, top_only=var_41)
    var_43 = list(var_42)

def test_case_15():
    var_0 = 'import os'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'import numpy as np'
    var_6 = [var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = 'from collections import defaultdict'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 1
    var_15 = 'from pathlib import Path as P'
    var_16 = [var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    assert var_19 == 1
    var_20 = 'import sys, os'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = len(var_23)
    assert var_24 == 2
    var_25 = 'cimport numpy'
    var_26 = [var_25]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = 'from typing import (\n    List,\n    Dict,\n)'
    var_31 = [var_30]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    var_34 = len(var_33)
    assert var_34 == 2
    var_35 = [var_4]
    var_36 = module_0.imports(var_35)
    with pytest.raises(TypeError):
        var_37 = list(var_36)

def test_case_16():
    var_0 = '\x0b4u2Q<dh:\\np{dm'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = list(var_2)
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = [var_0, var_3]
    var_6 = module_0.imports(var_5)
    var_7 = module_0.imports(var_1)
    var_8 = True
    var_9 = module_2.Config()
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
    assert var_9.atomic is False
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
    assert len(var_9.sources) == 1
    assert var_9.virtual_env == ''
    assert var_9.conda_env == ''
    assert var_9.ensure_newline_before_comments is False
    assert var_9.directory == '/workspace'
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_10 = 'import numpy as numpy\n'
    var_11 = [var_10]
    var_12 = module_0.imports(var_11, var_9)
    var_13 = list(var_12)
    var_14 = len(var_1)
    assert var_14 == 1
    var_15 = '    yield\n'
    var_16 = [var_15, var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    with pytest.raises(TypeError):
        var_19 = len(var_14)
    assert var_19 == 1

def test_case_17():
    var_0 = []
    var_1 = module_0.imports(var_0)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_2 = list(var_1)
    var_3 = 'from pandas import DaaFrame as DF\n'
    var_4 = [var_3]
    var_5 = module_0.imports(var_4)
    var_6 = 'import os, sys\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 'from os import path, environ\n'
    var_12 = [var_11, var_3]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    var_16 = 'from typing import (\n    List,\n    Dict,\n)$'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = '   impotos\n'
    var_21 = [var_20]
    var_22 = list(var_13)
    var_23 = len(var_22)
    var_24 = module_0.imports(var_21)
    var_25 = list(var_0)
    var_26 = len(var_25)
    var_27 = 'import os; import sys\n'
    var_28 = [var_27]
    var_29 = module_0.imports(var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    var_32 = True
    var_33 = module_2.Config()
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_34 = 'n$0X=PB\x0c\n\x0cJ8]F'
    var_35 = 'def foo():\n'
    var_36 = "'"
    var_37 = [var_19, var_35, var_36]
    var_38 = module_0.imports(var_37, top_only=var_32)
    var_39 = len(var_11)
    var_40 = module_0.imports(var_7)
    var_41 = list(var_40)
    var_42 = []
    var_43 = module_0.imports(var_42)
    var_44 = list(var_43)
    var_45 = list(var_2)
    var_46 = len(var_45)
    var_47 = 'raise ValueError\n'
    var_48 = [var_47, var_34]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)

def test_case_18():
    var_0 = 'import numpy as np\n'
    var_1 = len(var_0)
    var_2 = 'from pandas import DaaFrame as DF\n'
    var_3 = 'fr`m os import path, en\\iron\n'
    var_4 = [var_3, var_2]
    var_5 = module_0.imports(var_4)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_6 = list(var_5)
    var_7 = len(var_6)
    var_8 = var_1.__repr__()
    assert var_8 == '19'
    var_9 = list(var_8)
    var_10 = '{etl'
    var_11 = [var_10]
    var_12 = len(var_3)
    var_13 = module_0.imports(var_11)
    var_14 = module_0.imports(var_8)
    var_15 = var_11.__repr__()
    assert var_15 == "['{etl']"
    var_16 = list(var_15)
    var_17 = '    yield\n'
    var_18 = [var_15, var_17, var_10]
    var_19 = module_0.imports(var_18)
    var_20 = list(var_19)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = None
    var_2 = module_2.Config(var_1)
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_3 = var_2.__eq__(var_0)
    var_4 = var_3.__repr__()
    assert var_4 == 'NotImplemented'
    var_5 = module_0.imports(var_4, var_4, var_4)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_6 = var_4.__repr__()
    assert var_6 == "'NotImplemented'"
    var_7 = [var_6, var_0, var_3, var_3]
    var_8 = var_2.__eq__(var_0)
    var_9 = module_0.Import(*var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.identify.Import'
    assert len(var_9) == 7
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_10 = var_9.statement()
    assert var_10 == 'from NotImplemented import NotImplemented'
    var_3.is_char_device()

def test_case_20():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = module_0.imports(var_2)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'import numpy as np\n'
    var_7 = [var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = len(var_9)
    assert var_10 == 1
    var_11 = 'from collections import defaultdict\n'
    var_12 = [var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = len(var_14)
    assert var_15 == 1
    var_16 = 'from pathlib import Path as P\n'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = len(var_19)
    assert var_20 == 1
    var_21 = 'from typing import List, Dict, Set\n'
    var_22 = [var_21]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)
    var_25 = len(var_24)
    assert var_25 == 3
    var_26 = 'cimport numpy\n'
    var_27 = [var_26]
    var_28 = module_0.imports(var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 1
    var_31 = 'from libc cimport malloc\n'
    var_32 = [var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = 'from collections import (\n'
    var_37 = '    defaultdict,\n'
    var_38 = '    OrderedDict,\n'
    var_39 = ')\n'
    var_40 = [var_36, var_37, var_38, var_39]
    var_41 = module_0.imports(var_40)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = '    import os\n'
    var_45 = [var_44]
    var_46 = module_0.imports(var_45)
    var_47 = list(var_46)
    var_48 = len(var_47)
    assert var_48 == 1
    var_49 = 'import os  # Operating system interfaces\n'
    var_50 = [var_49]
    var_51 = module_0.imports(var_50)
    var_52 = list(var_51)
    var_53 = len(var_52)
    assert var_53 == 1
    var_54 = True
    var_55 = module_2.Config()
    assert f'{type(var_55).__module__}.{type(var_55).__qualname__}' == 'isort.settings.Config'
    assert var_55.py_version == 'py3'
    assert f'{type(var_55.force_to_top).__module__}.{type(var_55.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.force_to_top) == 0
    assert f'{type(var_55.skip).__module__}.{type(var_55.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.skip) == 19
    assert f'{type(var_55.extend_skip).__module__}.{type(var_55.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.extend_skip) == 0
    assert f'{type(var_55.skip_glob).__module__}.{type(var_55.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.skip_glob) == 0
    assert f'{type(var_55.extend_skip_glob).__module__}.{type(var_55.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.extend_skip_glob) == 0
    assert var_55.skip_gitignore is False
    assert var_55.line_length == 79
    assert var_55.wrap_length == 0
    assert var_55.line_ending == ''
    assert var_55.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_55.no_sections is False
    assert f'{type(var_55.known_future_library).__module__}.{type(var_55.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.known_future_library) == 1
    assert f'{type(var_55.known_third_party).__module__}.{type(var_55.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.known_third_party) == 0
    assert f'{type(var_55.known_first_party).__module__}.{type(var_55.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.known_first_party) == 0
    assert f'{type(var_55.known_local_folder).__module__}.{type(var_55.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.known_local_folder) == 0
    assert f'{type(var_55.known_standard_library).__module__}.{type(var_55.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.known_standard_library) == 234
    assert f'{type(var_55.extra_standard_library).__module__}.{type(var_55.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.extra_standard_library) == 0
    assert var_55.known_other == {}
    assert var_55.multi_line_output == module_1.WrapModes.GRID
    assert var_55.forced_separate == ()
    assert var_55.indent == '    '
    assert var_55.comment_prefix == '  #'
    assert var_55.length_sort is False
    assert var_55.length_sort_straight is False
    assert f'{type(var_55.length_sort_sections).__module__}.{type(var_55.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.length_sort_sections) == 0
    assert f'{type(var_55.add_imports).__module__}.{type(var_55.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.add_imports) == 0
    assert f'{type(var_55.remove_imports).__module__}.{type(var_55.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.remove_imports) == 0
    assert var_55.append_only is False
    assert var_55.reverse_relative is False
    assert var_55.force_single_line is False
    assert var_55.single_line_exclusions == ()
    assert var_55.default_section == 'THIRDPARTY'
    assert var_55.import_headings == {}
    assert var_55.import_footers == {}
    assert var_55.balanced_wrapping is False
    assert var_55.use_parentheses is False
    assert var_55.order_by_type is True
    assert var_55.atomic is False
    assert var_55.lines_before_imports == -1
    assert var_55.lines_after_imports == -1
    assert var_55.lines_between_sections == 1
    assert var_55.lines_between_types == 0
    assert var_55.combine_as_imports is False
    assert var_55.combine_star is False
    assert var_55.include_trailing_comma is False
    assert var_55.from_first is False
    assert var_55.verbose is False
    assert var_55.quiet is False
    assert var_55.force_adds is False
    assert var_55.force_alphabetical_sort_within_sections is False
    assert var_55.force_alphabetical_sort is False
    assert var_55.force_grid_wrap == 0
    assert var_55.force_sort_within_sections is False
    assert var_55.lexicographical is False
    assert var_55.group_by_package is False
    assert var_55.ignore_whitespace is False
    assert f'{type(var_55.no_lines_before).__module__}.{type(var_55.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.no_lines_before) == 0
    assert var_55.no_inline_sort is False
    assert var_55.ignore_comments is False
    assert var_55.case_sensitive is False
    assert f'{type(var_55.sources).__module__}.{type(var_55.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_55.sources) == 1
    assert var_55.virtual_env == ''
    assert var_55.conda_env == ''
    assert var_55.ensure_newline_before_comments is False
    assert var_55.directory == '/workspace'
    assert var_55.profile == ''
    assert var_55.honor_noqa is False
    assert f'{type(var_55.src_paths).__module__}.{type(var_55.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_55.src_paths) == 2
    assert var_55.remove_redundant_aliases is False
    assert var_55.float_to_top is False
    assert var_55.filter_files is False
    assert var_55.formatter == ''
    assert var_55.formatting_function is None
    assert var_55.color_output is False
    assert f'{type(var_55.treat_comments_as_code).__module__}.{type(var_55.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.treat_comments_as_code) == 0
    assert var_55.treat_all_comments_as_code is False
    assert f'{type(var_55.supported_extensions).__module__}.{type(var_55.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.supported_extensions) == 4
    assert f'{type(var_55.blocked_extensions).__module__}.{type(var_55.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.blocked_extensions) == 1
    assert f'{type(var_55.constants).__module__}.{type(var_55.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.constants) == 0
    assert f'{type(var_55.classes).__module__}.{type(var_55.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.classes) == 0
    assert f'{type(var_55.variables).__module__}.{type(var_55.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.variables) == 0
    assert var_55.dedup_headings is False
    assert var_55.only_sections is False
    assert var_55.only_modified is False
    assert var_55.combine_straight_imports is False
    assert var_55.auto_identify_namespace_packages is True
    assert f'{type(var_55.namespace_packages).__module__}.{type(var_55.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_55.namespace_packages) == 0
    assert var_55.follow_links is True
    assert var_55.indented_import_headings is True
    assert var_55.honor_case_in_force_sorted_sections is False
    assert var_55.sort_relative_in_force_sorted_sections is False
    assert var_55.overwrite_in_place is False
    assert var_55.reverse_sort is False
    assert var_55.star_first is False
    assert var_55.git_ls_files == {}
    assert var_55.format_error == '{error}: {message}'
    assert var_55.format_success == '{success}: {message}'
    assert var_55.sort_order == 'natural'
    assert var_55.sort_reexports is False
    assert var_55.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_56 = 'import os as os\n'
    var_57 = [var_56]
    var_58 = module_0.imports(var_57, var_55)
    var_59 = list(var_58)
    var_60 = len(var_59)
    assert var_60 == 1
    var_61 = len(var_59)
    assert var_61 == 1
    var_62 = 'def function():\n'
    var_63 = '    import sys\n'
    var_64 = [var_0, var_62, var_63]
    var_65 = module_0.imports(var_64, top_only=var_54)
    var_66 = list(var_65)
    var_67 = len(var_66)
    assert var_67 == 1

def test_case_21():
    var_0 = 'import numpy as np\n'
    var_1 = 'from typing import (\n'
    var_2 = [var_1, var_0, var_0, var_0]
    var_3 = module_0.imports(var_2)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_4 = list(var_3)
    var_5 = False
    with pytest.raises(TypeError):
        var_6 = list(var_5)

def test_case_22():
    var_0 = 'import os\n:\\'
    var_1 = [var_0, var_0, var_0, var_0]
    var_2 = [var_0]
    var_3 = 'from pathlib import Path as P\n'
    var_4 = [var_3]
    var_5 = module_0.imports(var_4)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_6 = list(var_3)
    var_7 = len(var_1)
    var_8 = 'cimport numy8'
    var_9 = list(var_1)
    var_10 = 'from typing import (\n'
    var_11 = list(var_5)
    var_12 = module_2.Config()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.settings.Config'
    assert var_12.py_version == 'py3'
    assert f'{type(var_12.force_to_top).__module__}.{type(var_12.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.force_to_top) == 0
    assert f'{type(var_12.skip).__module__}.{type(var_12.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.skip) == 19
    assert f'{type(var_12.extend_skip).__module__}.{type(var_12.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extend_skip) == 0
    assert f'{type(var_12.skip_glob).__module__}.{type(var_12.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.skip_glob) == 0
    assert f'{type(var_12.extend_skip_glob).__module__}.{type(var_12.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extend_skip_glob) == 0
    assert var_12.skip_gitignore is False
    assert var_12.line_length == 79
    assert var_12.wrap_length == 0
    assert var_12.line_ending == ''
    assert var_12.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_12.no_sections is False
    assert f'{type(var_12.known_future_library).__module__}.{type(var_12.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_future_library) == 1
    assert f'{type(var_12.known_third_party).__module__}.{type(var_12.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_third_party) == 0
    assert f'{type(var_12.known_first_party).__module__}.{type(var_12.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_first_party) == 0
    assert f'{type(var_12.known_local_folder).__module__}.{type(var_12.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_local_folder) == 0
    assert f'{type(var_12.known_standard_library).__module__}.{type(var_12.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.known_standard_library) == 234
    assert f'{type(var_12.extra_standard_library).__module__}.{type(var_12.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.extra_standard_library) == 0
    assert var_12.known_other == {}
    assert var_12.multi_line_output == module_1.WrapModes.GRID
    assert var_12.forced_separate == ()
    assert var_12.indent == '    '
    assert var_12.comment_prefix == '  #'
    assert var_12.length_sort is False
    assert var_12.length_sort_straight is False
    assert f'{type(var_12.length_sort_sections).__module__}.{type(var_12.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.length_sort_sections) == 0
    assert f'{type(var_12.add_imports).__module__}.{type(var_12.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.add_imports) == 0
    assert f'{type(var_12.remove_imports).__module__}.{type(var_12.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.remove_imports) == 0
    assert var_12.append_only is False
    assert var_12.reverse_relative is False
    assert var_12.force_single_line is False
    assert var_12.single_line_exclusions == ()
    assert var_12.default_section == 'THIRDPARTY'
    assert var_12.import_headings == {}
    assert var_12.import_footers == {}
    assert var_12.balanced_wrapping is False
    assert var_12.use_parentheses is False
    assert var_12.order_by_type is True
    assert var_12.atomic is False
    assert var_12.lines_before_imports == -1
    assert var_12.lines_after_imports == -1
    assert var_12.lines_between_sections == 1
    assert var_12.lines_between_types == 0
    assert var_12.combine_as_imports is False
    assert var_12.combine_star is False
    assert var_12.include_trailing_comma is False
    assert var_12.from_first is False
    assert var_12.verbose is False
    assert var_12.quiet is False
    assert var_12.force_adds is False
    assert var_12.force_alphabetical_sort_within_sections is False
    assert var_12.force_alphabetical_sort is False
    assert var_12.force_grid_wrap == 0
    assert var_12.force_sort_within_sections is False
    assert var_12.lexicographical is False
    assert var_12.group_by_package is False
    assert var_12.ignore_whitespace is False
    assert f'{type(var_12.no_lines_before).__module__}.{type(var_12.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.no_lines_before) == 0
    assert var_12.no_inline_sort is False
    assert var_12.ignore_comments is False
    assert var_12.case_sensitive is False
    assert f'{type(var_12.sources).__module__}.{type(var_12.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_12.sources) == 1
    assert var_12.virtual_env == ''
    assert var_12.conda_env == ''
    assert var_12.ensure_newline_before_comments is False
    assert var_12.directory == '/workspace'
    assert var_12.profile == ''
    assert var_12.honor_noqa is False
    assert f'{type(var_12.src_paths).__module__}.{type(var_12.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_12.src_paths) == 2
    assert var_12.remove_redundant_aliases is False
    assert var_12.float_to_top is False
    assert var_12.filter_files is False
    assert var_12.formatter == ''
    assert var_12.formatting_function is None
    assert var_12.color_output is False
    assert f'{type(var_12.treat_comments_as_code).__module__}.{type(var_12.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.treat_comments_as_code) == 0
    assert var_12.treat_all_comments_as_code is False
    assert f'{type(var_12.supported_extensions).__module__}.{type(var_12.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.supported_extensions) == 4
    assert f'{type(var_12.blocked_extensions).__module__}.{type(var_12.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.blocked_extensions) == 1
    assert f'{type(var_12.constants).__module__}.{type(var_12.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.constants) == 0
    assert f'{type(var_12.classes).__module__}.{type(var_12.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.classes) == 0
    assert f'{type(var_12.variables).__module__}.{type(var_12.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.variables) == 0
    assert var_12.dedup_headings is False
    assert var_12.only_sections is False
    assert var_12.only_modified is False
    assert var_12.combine_straight_imports is False
    assert var_12.auto_identify_namespace_packages is True
    assert f'{type(var_12.namespace_packages).__module__}.{type(var_12.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_12.namespace_packages) == 0
    assert var_12.follow_links is True
    assert var_12.indented_import_headings is True
    assert var_12.honor_case_in_force_sorted_sections is False
    assert var_12.sort_relative_in_force_sorted_sections is False
    assert var_12.overwrite_in_place is False
    assert var_12.reverse_sort is False
    assert var_12.star_first is False
    assert var_12.git_ls_files == {}
    assert var_12.format_error == '{error}: {message}'
    assert var_12.format_success == '{success}: {message}'
    assert var_12.sort_order == 'natural'
    assert var_12.sort_reexports is False
    assert var_12.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_13 = '/test/file.py'
    var_14 = module_0.imports(var_2, top_only=var_11)
    var_15 = list(var_14)
    var_16 = len(var_15)
    var_17 = list(var_11)
    var_18 = 'from typing import \\\n'
    var_19 = [var_18, var_10, var_18, var_8, var_13]
    var_20 = module_0.imports(var_19)
    var_21 = list(var_20)

def test_case_23():
    var_0 = 'import os\n:\\'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_0, var_0, var_1]
    var_3 = [var_0]
    var_4 = 'from pathlib import Path as P\n'
    var_5 = [var_4]
    var_6 = module_0.imports(var_5)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_7 = list(var_4)
    var_8 = len(var_2)
    var_9 = 'cimport numy8'
    var_10 = list(var_2)
    var_11 = 'from typing import (\n'
    var_12 = module_0.imports(var_2)
    var_13 = list(var_12)
    var_14 = list(var_6)
    var_15 = module_2.Config()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.settings.Config'
    assert var_15.py_version == 'py3'
    assert f'{type(var_15.force_to_top).__module__}.{type(var_15.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.force_to_top) == 0
    assert f'{type(var_15.skip).__module__}.{type(var_15.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.skip) == 19
    assert f'{type(var_15.extend_skip).__module__}.{type(var_15.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extend_skip) == 0
    assert f'{type(var_15.skip_glob).__module__}.{type(var_15.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.skip_glob) == 0
    assert f'{type(var_15.extend_skip_glob).__module__}.{type(var_15.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extend_skip_glob) == 0
    assert var_15.skip_gitignore is False
    assert var_15.line_length == 79
    assert var_15.wrap_length == 0
    assert var_15.line_ending == ''
    assert var_15.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_15.no_sections is False
    assert f'{type(var_15.known_future_library).__module__}.{type(var_15.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_future_library) == 1
    assert f'{type(var_15.known_third_party).__module__}.{type(var_15.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_third_party) == 0
    assert f'{type(var_15.known_first_party).__module__}.{type(var_15.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_first_party) == 0
    assert f'{type(var_15.known_local_folder).__module__}.{type(var_15.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_local_folder) == 0
    assert f'{type(var_15.known_standard_library).__module__}.{type(var_15.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_standard_library) == 234
    assert f'{type(var_15.extra_standard_library).__module__}.{type(var_15.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extra_standard_library) == 0
    assert var_15.known_other == {}
    assert var_15.multi_line_output == module_1.WrapModes.GRID
    assert var_15.forced_separate == ()
    assert var_15.indent == '    '
    assert var_15.comment_prefix == '  #'
    assert var_15.length_sort is False
    assert var_15.length_sort_straight is False
    assert f'{type(var_15.length_sort_sections).__module__}.{type(var_15.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.length_sort_sections) == 0
    assert f'{type(var_15.add_imports).__module__}.{type(var_15.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.add_imports) == 0
    assert f'{type(var_15.remove_imports).__module__}.{type(var_15.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.remove_imports) == 0
    assert var_15.append_only is False
    assert var_15.reverse_relative is False
    assert var_15.force_single_line is False
    assert var_15.single_line_exclusions == ()
    assert var_15.default_section == 'THIRDPARTY'
    assert var_15.import_headings == {}
    assert var_15.import_footers == {}
    assert var_15.balanced_wrapping is False
    assert var_15.use_parentheses is False
    assert var_15.order_by_type is True
    assert var_15.atomic is False
    assert var_15.lines_before_imports == -1
    assert var_15.lines_after_imports == -1
    assert var_15.lines_between_sections == 1
    assert var_15.lines_between_types == 0
    assert var_15.combine_as_imports is False
    assert var_15.combine_star is False
    assert var_15.include_trailing_comma is False
    assert var_15.from_first is False
    assert var_15.verbose is False
    assert var_15.quiet is False
    assert var_15.force_adds is False
    assert var_15.force_alphabetical_sort_within_sections is False
    assert var_15.force_alphabetical_sort is False
    assert var_15.force_grid_wrap == 0
    assert var_15.force_sort_within_sections is False
    assert var_15.lexicographical is False
    assert var_15.group_by_package is False
    assert var_15.ignore_whitespace is False
    assert f'{type(var_15.no_lines_before).__module__}.{type(var_15.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.no_lines_before) == 0
    assert var_15.no_inline_sort is False
    assert var_15.ignore_comments is False
    assert var_15.case_sensitive is False
    assert f'{type(var_15.sources).__module__}.{type(var_15.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_15.sources) == 1
    assert var_15.virtual_env == ''
    assert var_15.conda_env == ''
    assert var_15.ensure_newline_before_comments is False
    assert var_15.directory == '/workspace'
    assert var_15.profile == ''
    assert var_15.honor_noqa is False
    assert f'{type(var_15.src_paths).__module__}.{type(var_15.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_15.src_paths) == 2
    assert var_15.remove_redundant_aliases is False
    assert var_15.float_to_top is False
    assert var_15.filter_files is False
    assert var_15.formatter == ''
    assert var_15.formatting_function is None
    assert var_15.color_output is False
    assert f'{type(var_15.treat_comments_as_code).__module__}.{type(var_15.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.treat_comments_as_code) == 0
    assert var_15.treat_all_comments_as_code is False
    assert f'{type(var_15.supported_extensions).__module__}.{type(var_15.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.supported_extensions) == 4
    assert f'{type(var_15.blocked_extensions).__module__}.{type(var_15.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.blocked_extensions) == 1
    assert f'{type(var_15.constants).__module__}.{type(var_15.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.constants) == 0
    assert f'{type(var_15.classes).__module__}.{type(var_15.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.classes) == 0
    assert f'{type(var_15.variables).__module__}.{type(var_15.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.variables) == 0
    assert var_15.dedup_headings is False
    assert var_15.only_sections is False
    assert var_15.only_modified is False
    assert var_15.combine_straight_imports is False
    assert var_15.auto_identify_namespace_packages is True
    assert f'{type(var_15.namespace_packages).__module__}.{type(var_15.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.namespace_packages) == 0
    assert var_15.follow_links is True
    assert var_15.indented_import_headings is True
    assert var_15.honor_case_in_force_sorted_sections is False
    assert var_15.sort_relative_in_force_sorted_sections is False
    assert var_15.overwrite_in_place is False
    assert var_15.reverse_sort is False
    assert var_15.star_first is False
    assert var_15.git_ls_files == {}
    assert var_15.format_error == '{error}: {message}'
    assert var_15.format_success == '{success}: {message}'
    assert var_15.sort_order == 'natural'
    assert var_15.sort_reexports is False
    assert var_15.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_16 = '/test/file.py'
    var_17 = module_0.imports(var_3, top_only=var_12)
    var_18 = list(var_17)
    var_19 = len(var_18)
    var_20 = list(var_14)
    var_21 = 'from typing import \\\n'
    var_22 = [var_21, var_11, var_21, var_9, var_16]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)

def test_case_24():
    var_0 = 'import os\n:\\'
    var_1 = 'iort sys\n'
    var_2 = [var_0, var_0, var_0, var_1]
    var_3 = [var_0]
    var_4 = 'from pathlib import Path as P\n'
    var_5 = [var_4]
    var_6 = module_0.imports(var_5)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_7 = list(var_4)
    var_8 = len(var_2)
    var_9 = 'cimport numy8'
    var_10 = list(var_2)
    var_11 = 'from typing import (\n'
    var_12 = module_0.imports(var_2)
    var_13 = list(var_12)
    var_14 = list(var_6)
    var_15 = module_2.Config()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.settings.Config'
    assert var_15.py_version == 'py3'
    assert f'{type(var_15.force_to_top).__module__}.{type(var_15.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.force_to_top) == 0
    assert f'{type(var_15.skip).__module__}.{type(var_15.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.skip) == 19
    assert f'{type(var_15.extend_skip).__module__}.{type(var_15.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extend_skip) == 0
    assert f'{type(var_15.skip_glob).__module__}.{type(var_15.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.skip_glob) == 0
    assert f'{type(var_15.extend_skip_glob).__module__}.{type(var_15.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extend_skip_glob) == 0
    assert var_15.skip_gitignore is False
    assert var_15.line_length == 79
    assert var_15.wrap_length == 0
    assert var_15.line_ending == ''
    assert var_15.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_15.no_sections is False
    assert f'{type(var_15.known_future_library).__module__}.{type(var_15.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_future_library) == 1
    assert f'{type(var_15.known_third_party).__module__}.{type(var_15.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_third_party) == 0
    assert f'{type(var_15.known_first_party).__module__}.{type(var_15.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_first_party) == 0
    assert f'{type(var_15.known_local_folder).__module__}.{type(var_15.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_local_folder) == 0
    assert f'{type(var_15.known_standard_library).__module__}.{type(var_15.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.known_standard_library) == 234
    assert f'{type(var_15.extra_standard_library).__module__}.{type(var_15.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.extra_standard_library) == 0
    assert var_15.known_other == {}
    assert var_15.multi_line_output == module_1.WrapModes.GRID
    assert var_15.forced_separate == ()
    assert var_15.indent == '    '
    assert var_15.comment_prefix == '  #'
    assert var_15.length_sort is False
    assert var_15.length_sort_straight is False
    assert f'{type(var_15.length_sort_sections).__module__}.{type(var_15.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.length_sort_sections) == 0
    assert f'{type(var_15.add_imports).__module__}.{type(var_15.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.add_imports) == 0
    assert f'{type(var_15.remove_imports).__module__}.{type(var_15.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.remove_imports) == 0
    assert var_15.append_only is False
    assert var_15.reverse_relative is False
    assert var_15.force_single_line is False
    assert var_15.single_line_exclusions == ()
    assert var_15.default_section == 'THIRDPARTY'
    assert var_15.import_headings == {}
    assert var_15.import_footers == {}
    assert var_15.balanced_wrapping is False
    assert var_15.use_parentheses is False
    assert var_15.order_by_type is True
    assert var_15.atomic is False
    assert var_15.lines_before_imports == -1
    assert var_15.lines_after_imports == -1
    assert var_15.lines_between_sections == 1
    assert var_15.lines_between_types == 0
    assert var_15.combine_as_imports is False
    assert var_15.combine_star is False
    assert var_15.include_trailing_comma is False
    assert var_15.from_first is False
    assert var_15.verbose is False
    assert var_15.quiet is False
    assert var_15.force_adds is False
    assert var_15.force_alphabetical_sort_within_sections is False
    assert var_15.force_alphabetical_sort is False
    assert var_15.force_grid_wrap == 0
    assert var_15.force_sort_within_sections is False
    assert var_15.lexicographical is False
    assert var_15.group_by_package is False
    assert var_15.ignore_whitespace is False
    assert f'{type(var_15.no_lines_before).__module__}.{type(var_15.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.no_lines_before) == 0
    assert var_15.no_inline_sort is False
    assert var_15.ignore_comments is False
    assert var_15.case_sensitive is False
    assert f'{type(var_15.sources).__module__}.{type(var_15.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_15.sources) == 1
    assert var_15.virtual_env == ''
    assert var_15.conda_env == ''
    assert var_15.ensure_newline_before_comments is False
    assert var_15.directory == '/workspace'
    assert var_15.profile == ''
    assert var_15.honor_noqa is False
    assert f'{type(var_15.src_paths).__module__}.{type(var_15.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_15.src_paths) == 2
    assert var_15.remove_redundant_aliases is False
    assert var_15.float_to_top is False
    assert var_15.filter_files is False
    assert var_15.formatter == ''
    assert var_15.formatting_function is None
    assert var_15.color_output is False
    assert f'{type(var_15.treat_comments_as_code).__module__}.{type(var_15.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.treat_comments_as_code) == 0
    assert var_15.treat_all_comments_as_code is False
    assert f'{type(var_15.supported_extensions).__module__}.{type(var_15.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.supported_extensions) == 4
    assert f'{type(var_15.blocked_extensions).__module__}.{type(var_15.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.blocked_extensions) == 1
    assert f'{type(var_15.constants).__module__}.{type(var_15.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.constants) == 0
    assert f'{type(var_15.classes).__module__}.{type(var_15.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.classes) == 0
    assert f'{type(var_15.variables).__module__}.{type(var_15.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.variables) == 0
    assert var_15.dedup_headings is False
    assert var_15.only_sections is False
    assert var_15.only_modified is False
    assert var_15.combine_straight_imports is False
    assert var_15.auto_identify_namespace_packages is True
    assert f'{type(var_15.namespace_packages).__module__}.{type(var_15.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.namespace_packages) == 0
    assert var_15.follow_links is True
    assert var_15.indented_import_headings is True
    assert var_15.honor_case_in_force_sorted_sections is False
    assert var_15.sort_relative_in_force_sorted_sections is False
    assert var_15.overwrite_in_place is False
    assert var_15.reverse_sort is False
    assert var_15.star_first is False
    assert var_15.git_ls_files == {}
    assert var_15.format_error == '{error}: {message}'
    assert var_15.format_success == '{success}: {message}'
    assert var_15.sort_order == 'natural'
    assert var_15.sort_reexports is False
    assert var_15.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_16 = '/test/file.py'
    var_17 = module_0.imports(var_3, top_only=var_12)
    var_18 = list(var_17)
    var_19 = len(var_18)
    var_20 = list(var_14)
    var_21 = 'from typing import \\\n'
    var_22 = [var_21, var_11, var_21, var_9, var_16]
    var_23 = module_0.imports(var_22)
    var_24 = list(var_23)

def test_case_25():
    var_0 = 'import numpy as np\n'
    var_1 = [var_0]
    var_2 = module_0.imports(var_1)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_3 = 'from pandas import DaaFrame as DF\n'
    var_4 = 'cimport numpy\n'
    var_5 = 'import os, sys\n'
    var_6 = [var_5]
    var_7 = list(var_2)
    var_8 = len(var_7)
    var_9 = 'from os import path, environ\n'
    var_10 = [var_9, var_3]
    var_11 = module_0.imports(var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    var_14 = 'from typing import (\n    List,\n    Dict,\n)$'
    var_15 = [var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = ''
    var_19 = list(var_4)
    var_20 = len(var_19)
    var_21 = module_0.imports(var_10)
    var_22 = list(var_2)
    var_23 = len(var_4)
    var_24 = [var_17, var_7, var_11]
    var_25 = module_0.imports(var_24, top_only=var_12)
    var_26 = len(var_9)
    var_27 = list(var_19)
    var_28 = list(var_6)
    var_29 = len(var_28)
    assert var_29 == 1
    var_30 = '    yield\n'
    var_31 = [var_30, var_30, var_18]
    var_32 = module_0.imports(var_31)
    var_33 = list(var_32)
    with pytest.raises(AttributeError):
        var_34 = list(var_25)

def test_case_26():
    var_0 = 'import os\n'
    var_1 = 'import sys\n'
    var_2 = [var_0, var_1]
    var_3 = 'from pathlib import Path as P\n'
    var_4 = [var_3]
    var_5 = module_0.imports(var_4)
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
    assert module_0.STATEMENT_DECLARATIONS == ('def ', 'cdef ', 'cpdef ', 'class ', '@', 'async def')
    var_6 = list(var_4)
    var_7 = '2:Lh.wS75&A\rKu'
    var_8 = list(var_2)
    var_9 = 'from typing import (\n'
    var_10 = '    List,\n'
    var_11 = '    Dict,\n'
    var_12 = [var_9, var_10, var_11, var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = list(var_5)
    var_16 = len(var_15)
    assert var_16 == 1
    var_17 = module_2.Config()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'isort.settings.Config'
    assert var_17.py_version == 'py3'
    assert f'{type(var_17.force_to_top).__module__}.{type(var_17.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.force_to_top) == 0
    assert f'{type(var_17.skip).__module__}.{type(var_17.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.skip) == 19
    assert f'{type(var_17.extend_skip).__module__}.{type(var_17.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extend_skip) == 0
    assert f'{type(var_17.skip_glob).__module__}.{type(var_17.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.skip_glob) == 0
    assert f'{type(var_17.extend_skip_glob).__module__}.{type(var_17.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extend_skip_glob) == 0
    assert var_17.skip_gitignore is False
    assert var_17.line_length == 79
    assert var_17.wrap_length == 0
    assert var_17.line_ending == ''
    assert var_17.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_17.no_sections is False
    assert f'{type(var_17.known_future_library).__module__}.{type(var_17.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_future_library) == 1
    assert f'{type(var_17.known_third_party).__module__}.{type(var_17.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_third_party) == 0
    assert f'{type(var_17.known_first_party).__module__}.{type(var_17.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_first_party) == 0
    assert f'{type(var_17.known_local_folder).__module__}.{type(var_17.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_local_folder) == 0
    assert f'{type(var_17.known_standard_library).__module__}.{type(var_17.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.known_standard_library) == 234
    assert f'{type(var_17.extra_standard_library).__module__}.{type(var_17.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.extra_standard_library) == 0
    assert var_17.known_other == {}
    assert var_17.multi_line_output == module_1.WrapModes.GRID
    assert var_17.forced_separate == ()
    assert var_17.indent == '    '
    assert var_17.comment_prefix == '  #'
    assert var_17.length_sort is False
    assert var_17.length_sort_straight is False
    assert f'{type(var_17.length_sort_sections).__module__}.{type(var_17.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.length_sort_sections) == 0
    assert f'{type(var_17.add_imports).__module__}.{type(var_17.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.add_imports) == 0
    assert f'{type(var_17.remove_imports).__module__}.{type(var_17.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.remove_imports) == 0
    assert var_17.append_only is False
    assert var_17.reverse_relative is False
    assert var_17.force_single_line is False
    assert var_17.single_line_exclusions == ()
    assert var_17.default_section == 'THIRDPARTY'
    assert var_17.import_headings == {}
    assert var_17.import_footers == {}
    assert var_17.balanced_wrapping is False
    assert var_17.use_parentheses is False
    assert var_17.order_by_type is True
    assert var_17.atomic is False
    assert var_17.lines_before_imports == -1
    assert var_17.lines_after_imports == -1
    assert var_17.lines_between_sections == 1
    assert var_17.lines_between_types == 0
    assert var_17.combine_as_imports is False
    assert var_17.combine_star is False
    assert var_17.include_trailing_comma is False
    assert var_17.from_first is False
    assert var_17.verbose is False
    assert var_17.quiet is False
    assert var_17.force_adds is False
    assert var_17.force_alphabetical_sort_within_sections is False
    assert var_17.force_alphabetical_sort is False
    assert var_17.force_grid_wrap == 0
    assert var_17.force_sort_within_sections is False
    assert var_17.lexicographical is False
    assert var_17.group_by_package is False
    assert var_17.ignore_whitespace is False
    assert f'{type(var_17.no_lines_before).__module__}.{type(var_17.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.no_lines_before) == 0
    assert var_17.no_inline_sort is False
    assert var_17.ignore_comments is False
    assert var_17.case_sensitive is False
    assert f'{type(var_17.sources).__module__}.{type(var_17.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_17.sources) == 1
    assert var_17.virtual_env == ''
    assert var_17.conda_env == ''
    assert var_17.ensure_newline_before_comments is False
    assert var_17.directory == '/workspace'
    assert var_17.profile == ''
    assert var_17.honor_noqa is False
    assert f'{type(var_17.src_paths).__module__}.{type(var_17.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_17.src_paths) == 2
    assert var_17.remove_redundant_aliases is False
    assert var_17.float_to_top is False
    assert var_17.filter_files is False
    assert var_17.formatter == ''
    assert var_17.formatting_function is None
    assert var_17.color_output is False
    assert f'{type(var_17.treat_comments_as_code).__module__}.{type(var_17.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.treat_comments_as_code) == 0
    assert var_17.treat_all_comments_as_code is False
    assert f'{type(var_17.supported_extensions).__module__}.{type(var_17.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.supported_extensions) == 4
    assert f'{type(var_17.blocked_extensions).__module__}.{type(var_17.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.blocked_extensions) == 1
    assert f'{type(var_17.constants).__module__}.{type(var_17.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.constants) == 0
    assert f'{type(var_17.classes).__module__}.{type(var_17.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.classes) == 0
    assert f'{type(var_17.variables).__module__}.{type(var_17.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.variables) == 0
    assert var_17.dedup_headings is False
    assert var_17.only_sections is False
    assert var_17.only_modified is False
    assert var_17.combine_straight_imports is False
    assert var_17.auto_identify_namespace_packages is True
    assert f'{type(var_17.namespace_packages).__module__}.{type(var_17.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.namespace_packages) == 0
    assert var_17.follow_links is True
    assert var_17.indented_import_headings is True
    assert var_17.honor_case_in_force_sorted_sections is False
    assert var_17.sort_relative_in_force_sorted_sections is False
    assert var_17.overwrite_in_place is False
    assert var_17.reverse_sort is False
    assert var_17.star_first is False
    assert var_17.git_ls_files == {}
    assert var_17.format_error == '{error}: {message}'
    assert var_17.format_success == '{success}: {message}'
    assert var_17.sort_order == 'natural'
    assert var_17.sort_reexports is False
    assert var_17.split_on_trailing_comma is False
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
    assert module_2.DEFAULT_CONFIG.directory == '/workspace'
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
    var_18 = 'import numpy as numpy\n'
    var_19 = [var_18]
    var_20 = module_0.imports(var_19, var_17)
    var_21 = list(var_20)
    var_22 = len(var_8)
    var_23 = '    import sys\n'
    var_24 = [var_23]
    var_25 = module_0.imports(var_24)
    var_26 = len(var_23)
    var_27 = '/test/file.py'
    var_28 = [var_0]
    var_29 = module_0.imports(var_28, top_only=var_13)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = list(var_15)
    var_33 = len(var_32)
    assert var_33 == 1
    var_34 = [var_18, var_9, var_18, var_7, var_27]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)