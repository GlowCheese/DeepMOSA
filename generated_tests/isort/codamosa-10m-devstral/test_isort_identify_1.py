# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.identify as module_0
import isort.wrap_modes as module_1
import isort.parse as module_2
import isort.settings as module_3

def test_case_0():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0, var_0]
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
    var_1 = [var_0, var_0, var_0, var_0, var_0, var_0]
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
    var_0 = 'import numpy(as np\n'
    var_1 = [var_0, var_0, var_0]
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
    var_4 = iter(var_2)
    var_5 = list(var_3)

def test_case_4():
    var_0 = 'import numpy as np\n'
    var_1 = module_2.normalize_line(var_0)
    assert module_2.TYPE_CHECKING is False
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

def test_case_5():
    var_0 = True
    var_1 = 'DataFrame'
    var_2 = 'from sys import argv as argv'
    var_3 = [var_1, var_2]
    var_4 = module_3.Config()
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
    var_5 = module_0.imports(var_3, var_4)
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

def test_case_6():
    var_0 = 'import numpy as np\n'
    var_1 = iter(var_0)
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
    var_4 = iter(var_1)

def test_case_7():
    var_0 = 'Yrom os |mportpayh'
    var_1 = 'l\rah,Lf.,_B$x'
    var_2 = 'DataFrame'
    var_3 = 'int'
    var_4 = 'from os import ('
    var_5 = [var_4, var_2, var_0, var_3]
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
    var_7 = list(var_6)
    var_8 = list(var_6)
    var_9 = 'import os as os'
    var_10 = [var_9, var_1]
    var_11 = True
    var_12 = module_3.Config()
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
    var_13 = module_0.imports(var_10, var_12)
    var_14 = list(var_13)

def test_case_8():
    var_0 = 'import os'
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
    var_3 = list(var_2)
    var_4 = 'Yrom os |mportpayh'
    var_5 = 'from sys import argv'
    var_6 = [var_4, var_5]
    var_7 = module_0.imports(var_6)
    var_8 = list(var_7)
    var_9 = 'import numpy as np'
    var_10 = 'from pandas import DataFrame as df'
    var_11 = [var_9, var_10]
    var_12 = module_0.imports(var_11)
    var_13 = list(var_12)
    var_14 = 'from cython cimport int'
    var_15 = [var_4, var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = 'from os import ('
    var_19 = '  Rpat,'
    var_20 = '    environ'
    var_21 = [var_18, var_19, var_20, var_9]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = list(var_13)
    var_25 = '# This is a comment'
    var_26 = 'import os  # inline comment'
    var_27 = "'''docstring'''"
    var_28 = [var_25, var_26, var_27, var_0]
    var_29 = module_0.imports(var_28)
    var_30 = list(var_29)
    var_31 = 'import os as os'
    var_32 = 'from sys import argv as argv'
    var_33 = [var_31, var_32]
    var_34 = False
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
    var_36 = module_0.imports(var_33, var_35)
    var_37 = list(var_36)
    var_38 = True
    var_39 = module_0.imports(var_33, top_only=var_38)
    var_40 = list(var_39)

def test_case_9():
    var_0 = 'bN]lK+[xt+p{.7G'
    var_1 = [var_0, var_0]
    var_2 = 'sys'
    var_3 = module_0.imports(var_1)
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
    var_5 = module_0.imports(var_4)
    var_6 = list(var_5)
    var_7 = 'import numpy as np'
    var_8 = 'from pandas import DataFrame as df'
    var_9 = [var_7, var_8]
    var_10 = 'DataFrame'
    var_11 = module_0.imports(var_9)
    var_12 = list(var_11)
    var_13 = 'cimport numpy'
    var_14 = 'from cython cimport int'
    var_15 = [var_13, var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = 'from os import ('
    var_19 = '    environ'
    var_20 = ')'
    var_21 = [var_18, var_10, var_19, var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = '    import os'
    var_25 = [var_24, var_0]
    var_26 = module_0.imports(var_25)
    var_27 = list(var_5)
    var_28 = 'import os as os'
    var_29 = [var_28, var_2]
    var_30 = True
    var_31 = module_3.Config()
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'isort.settings.Config'
    assert var_31.py_version == 'py3'
    assert f'{type(var_31.force_to_top).__module__}.{type(var_31.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.force_to_top) == 0
    assert f'{type(var_31.skip).__module__}.{type(var_31.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.skip) == 19
    assert f'{type(var_31.extend_skip).__module__}.{type(var_31.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.extend_skip) == 0
    assert f'{type(var_31.skip_glob).__module__}.{type(var_31.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.skip_glob) == 0
    assert f'{type(var_31.extend_skip_glob).__module__}.{type(var_31.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.extend_skip_glob) == 0
    assert var_31.skip_gitignore is False
    assert var_31.line_length == 79
    assert var_31.wrap_length == 0
    assert var_31.line_ending == ''
    assert var_31.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_31.no_sections is False
    assert f'{type(var_31.known_future_library).__module__}.{type(var_31.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.known_future_library) == 1
    assert f'{type(var_31.known_third_party).__module__}.{type(var_31.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.known_third_party) == 0
    assert f'{type(var_31.known_first_party).__module__}.{type(var_31.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.known_first_party) == 0
    assert f'{type(var_31.known_local_folder).__module__}.{type(var_31.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.known_local_folder) == 0
    assert f'{type(var_31.known_standard_library).__module__}.{type(var_31.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.known_standard_library) == 234
    assert f'{type(var_31.extra_standard_library).__module__}.{type(var_31.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.extra_standard_library) == 0
    assert var_31.known_other == {}
    assert var_31.multi_line_output == module_1.WrapModes.GRID
    assert var_31.forced_separate == ()
    assert var_31.indent == '    '
    assert var_31.comment_prefix == '  #'
    assert var_31.length_sort is False
    assert var_31.length_sort_straight is False
    assert f'{type(var_31.length_sort_sections).__module__}.{type(var_31.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.length_sort_sections) == 0
    assert f'{type(var_31.add_imports).__module__}.{type(var_31.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.add_imports) == 0
    assert f'{type(var_31.remove_imports).__module__}.{type(var_31.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.remove_imports) == 0
    assert var_31.append_only is False
    assert var_31.reverse_relative is False
    assert var_31.force_single_line is False
    assert var_31.single_line_exclusions == ()
    assert var_31.default_section == 'THIRDPARTY'
    assert var_31.import_headings == {}
    assert var_31.import_footers == {}
    assert var_31.balanced_wrapping is False
    assert var_31.use_parentheses is False
    assert var_31.order_by_type is True
    assert var_31.atomic is False
    assert var_31.lines_before_imports == -1
    assert var_31.lines_after_imports == -1
    assert var_31.lines_between_sections == 1
    assert var_31.lines_between_types == 0
    assert var_31.combine_as_imports is False
    assert var_31.combine_star is False
    assert var_31.include_trailing_comma is False
    assert var_31.from_first is False
    assert var_31.verbose is False
    assert var_31.quiet is False
    assert var_31.force_adds is False
    assert var_31.force_alphabetical_sort_within_sections is False
    assert var_31.force_alphabetical_sort is False
    assert var_31.force_grid_wrap == 0
    assert var_31.force_sort_within_sections is False
    assert var_31.lexicographical is False
    assert var_31.group_by_package is False
    assert var_31.ignore_whitespace is False
    assert f'{type(var_31.no_lines_before).__module__}.{type(var_31.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.no_lines_before) == 0
    assert var_31.no_inline_sort is False
    assert var_31.ignore_comments is False
    assert var_31.case_sensitive is False
    assert f'{type(var_31.sources).__module__}.{type(var_31.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_31.sources) == 1
    assert var_31.virtual_env == ''
    assert var_31.conda_env == ''
    assert var_31.ensure_newline_before_comments is False
    assert var_31.directory == '/workspace'
    assert var_31.profile == ''
    assert var_31.honor_noqa is False
    assert f'{type(var_31.src_paths).__module__}.{type(var_31.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_31.src_paths) == 2
    assert var_31.remove_redundant_aliases is False
    assert var_31.float_to_top is False
    assert var_31.filter_files is False
    assert var_31.formatter == ''
    assert var_31.formatting_function is None
    assert var_31.color_output is False
    assert f'{type(var_31.treat_comments_as_code).__module__}.{type(var_31.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.treat_comments_as_code) == 0
    assert var_31.treat_all_comments_as_code is False
    assert f'{type(var_31.supported_extensions).__module__}.{type(var_31.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.supported_extensions) == 4
    assert f'{type(var_31.blocked_extensions).__module__}.{type(var_31.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.blocked_extensions) == 1
    assert f'{type(var_31.constants).__module__}.{type(var_31.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.constants) == 0
    assert f'{type(var_31.classes).__module__}.{type(var_31.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.classes) == 0
    assert f'{type(var_31.variables).__module__}.{type(var_31.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.variables) == 0
    assert var_31.dedup_headings is False
    assert var_31.only_sections is False
    assert var_31.only_modified is False
    assert var_31.combine_straight_imports is False
    assert var_31.auto_identify_namespace_packages is True
    assert f'{type(var_31.namespace_packages).__module__}.{type(var_31.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_31.namespace_packages) == 0
    assert var_31.follow_links is True
    assert var_31.indented_import_headings is True
    assert var_31.honor_case_in_force_sorted_sections is False
    assert var_31.sort_relative_in_force_sorted_sections is False
    assert var_31.overwrite_in_place is False
    assert var_31.reverse_sort is False
    assert var_31.star_first is False
    assert var_31.git_ls_files == {}
    assert var_31.format_error == '{error}: {message}'
    assert var_31.format_success == '{success}: {message}'
    assert var_31.sort_order == 'natural'
    assert var_31.sort_reexports is False
    assert var_31.split_on_trailing_comma is False
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
    var_32 = module_0.imports(var_29, var_31)
    var_33 = list(var_32)
    var_34 = 'def foo():'
    var_35 = '    import sys'
    var_36 = [var_0, var_34, var_35]
    var_37 = True
    var_38 = module_0.imports(var_36, top_only=var_37)
    var_39 = list(var_38)

def test_case_10():
    var_0 = 'from os import path'
    var_1 = 'from sys import argv'
    var_2 = [var_0, var_1]
    var_3 = 'argv'
    var_4 = module_0.imports(var_2)
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
    var_5 = list(var_4)
    var_6 = 'from pandas import DataFrame as df'
    var_7 = [var_3, var_6]
    var_8 = 'df'
    var_9 = module_0.imports(var_7)
    var_10 = list(var_9)
    var_11 = 'from cython cimport int'
    var_12 = [var_8, var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = 'from os import ('
    var_16 = '    environ'
    var_17 = ')'
    var_18 = [var_15, var_10, var_16, var_17]
    var_19 = module_0.imports(var_18)
    with pytest.raises(AttributeError):
        var_20 = list(var_19)

def test_case_11():
    var_0 = 'import numpy  np\n'
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
    var_4 = iter(var_2)
    var_5 = iter(var_1)
    var_6 = list(var_5)

def test_case_12():
    var_0 = 'import numpy as np'
    var_1 = 'from os import ('
    var_2 = '  Rpat,'
    var_3 = '    environ'
    var_4 = [var_1, var_2, var_3, var_0]
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
    var_7 = 'bh>'
    var_8 = 'from sys import argv as argv'
    var_9 = [var_7, var_8]
    var_10 = module_0.imports(var_6)
    var_11 = True
    var_12 = module_3.Config()
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
    var_13 = module_0.imports(var_9, var_12)
    var_14 = list(var_13)

def test_case_13():
    var_0 = 'l\rah,Lf.,_B$x'
    var_1 = 'import os as os'
    var_2 = [var_1, var_0]
    var_3 = True
    var_4 = module_3.Config()
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
    var_5 = module_0.imports(var_2, var_4)
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

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'sy\ns'
    var_1 = 'Yrom os |mportpayh'
    var_2 = [var_1, var_0]
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
    var_5 = 'import numpy as np'
    var_6 = module_0.imports(var_4)
    var_7 = 'from cython cimport int'
    var_8 = [var_1, var_7]
    var_9 = 'int'
    var_10 = module_0.imports(var_8)
    var_11 = list(var_10)
    var_12 = '  Rpat,'
    var_13 = [var_7, var_12, var_7, var_5]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = list(var_3)
    var_17 = '# This is a comment'
    var_18 = 'import os  # inline comment'
    var_19 = "'''docstring'''"
    var_20 = [var_17, var_18, var_19, var_9]
    var_21 = module_0.imports(var_20)
    var_22 = list(var_21)
    var_23 = 'import os as os'
    var_24 = 'from sys import argv as argv'
    var_25 = [var_23, var_24]
    var_26 = True
    var_27 = module_3.Config()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'isort.settings.Config'
    assert var_27.py_version == 'py3'
    assert f'{type(var_27.force_to_top).__module__}.{type(var_27.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.force_to_top) == 0
    assert f'{type(var_27.skip).__module__}.{type(var_27.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip) == 19
    assert f'{type(var_27.extend_skip).__module__}.{type(var_27.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip) == 0
    assert f'{type(var_27.skip_glob).__module__}.{type(var_27.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip_glob) == 0
    assert f'{type(var_27.extend_skip_glob).__module__}.{type(var_27.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip_glob) == 0
    assert var_27.skip_gitignore is False
    assert var_27.line_length == 79
    assert var_27.wrap_length == 0
    assert var_27.line_ending == ''
    assert var_27.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_27.no_sections is False
    assert f'{type(var_27.known_future_library).__module__}.{type(var_27.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_future_library) == 1
    assert f'{type(var_27.known_third_party).__module__}.{type(var_27.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_third_party) == 0
    assert f'{type(var_27.known_first_party).__module__}.{type(var_27.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_first_party) == 0
    assert f'{type(var_27.known_local_folder).__module__}.{type(var_27.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_local_folder) == 0
    assert f'{type(var_27.known_standard_library).__module__}.{type(var_27.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_standard_library) == 234
    assert f'{type(var_27.extra_standard_library).__module__}.{type(var_27.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extra_standard_library) == 0
    assert var_27.known_other == {}
    assert var_27.multi_line_output == module_1.WrapModes.GRID
    assert var_27.forced_separate == ()
    assert var_27.indent == '    '
    assert var_27.comment_prefix == '  #'
    assert var_27.length_sort is False
    assert var_27.length_sort_straight is False
    assert f'{type(var_27.length_sort_sections).__module__}.{type(var_27.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.length_sort_sections) == 0
    assert f'{type(var_27.add_imports).__module__}.{type(var_27.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.add_imports) == 0
    assert f'{type(var_27.remove_imports).__module__}.{type(var_27.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.remove_imports) == 0
    assert var_27.append_only is False
    assert var_27.reverse_relative is False
    assert var_27.force_single_line is False
    assert var_27.single_line_exclusions == ()
    assert var_27.default_section == 'THIRDPARTY'
    assert var_27.import_headings == {}
    assert var_27.import_footers == {}
    assert var_27.balanced_wrapping is False
    assert var_27.use_parentheses is False
    assert var_27.order_by_type is True
    assert var_27.atomic is False
    assert var_27.lines_before_imports == -1
    assert var_27.lines_after_imports == -1
    assert var_27.lines_between_sections == 1
    assert var_27.lines_between_types == 0
    assert var_27.combine_as_imports is False
    assert var_27.combine_star is False
    assert var_27.include_trailing_comma is False
    assert var_27.from_first is False
    assert var_27.verbose is False
    assert var_27.quiet is False
    assert var_27.force_adds is False
    assert var_27.force_alphabetical_sort_within_sections is False
    assert var_27.force_alphabetical_sort is False
    assert var_27.force_grid_wrap == 0
    assert var_27.force_sort_within_sections is False
    assert var_27.lexicographical is False
    assert var_27.group_by_package is False
    assert var_27.ignore_whitespace is False
    assert f'{type(var_27.no_lines_before).__module__}.{type(var_27.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.no_lines_before) == 0
    assert var_27.no_inline_sort is False
    assert var_27.ignore_comments is False
    assert var_27.case_sensitive is False
    assert f'{type(var_27.sources).__module__}.{type(var_27.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_27.sources) == 1
    assert var_27.virtual_env == ''
    assert var_27.conda_env == ''
    assert var_27.ensure_newline_before_comments is False
    assert var_27.directory == '/workspace'
    assert var_27.profile == ''
    assert var_27.honor_noqa is False
    assert f'{type(var_27.src_paths).__module__}.{type(var_27.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_27.src_paths) == 2
    assert var_27.remove_redundant_aliases is False
    assert var_27.float_to_top is False
    assert var_27.filter_files is False
    assert var_27.formatter == ''
    assert var_27.formatting_function is None
    assert var_27.color_output is False
    assert f'{type(var_27.treat_comments_as_code).__module__}.{type(var_27.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.treat_comments_as_code) == 0
    assert var_27.treat_all_comments_as_code is False
    assert f'{type(var_27.supported_extensions).__module__}.{type(var_27.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.supported_extensions) == 4
    assert f'{type(var_27.blocked_extensions).__module__}.{type(var_27.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.blocked_extensions) == 1
    assert f'{type(var_27.constants).__module__}.{type(var_27.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.constants) == 0
    assert f'{type(var_27.classes).__module__}.{type(var_27.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.classes) == 0
    assert f'{type(var_27.variables).__module__}.{type(var_27.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.variables) == 0
    assert var_27.dedup_headings is False
    assert var_27.only_sections is False
    assert var_27.only_modified is False
    assert var_27.combine_straight_imports is False
    assert var_27.auto_identify_namespace_packages is True
    assert f'{type(var_27.namespace_packages).__module__}.{type(var_27.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.namespace_packages) == 0
    assert var_27.follow_links is True
    assert var_27.indented_import_headings is True
    assert var_27.honor_case_in_force_sorted_sections is False
    assert var_27.sort_relative_in_force_sorted_sections is False
    assert var_27.overwrite_in_place is False
    assert var_27.reverse_sort is False
    assert var_27.star_first is False
    assert var_27.git_ls_files == {}
    assert var_27.format_error == '{error}: {message}'
    assert var_27.format_success == '{success}: {message}'
    assert var_27.sort_order == 'natural'
    assert var_27.sort_reexports is False
    assert var_27.split_on_trailing_comma is False
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
    var_28 = module_0.imports(var_25, var_27)
    var_29 = list(var_28)
    var_15.statement()

def test_case_15():
    var_0 = 'from os import path'
    var_1 = '}Jt`.gg<@~A+">`'
    var_2 = [var_0, var_1]
    var_3 = 'argv'
    var_4 = module_0.imports(var_2)
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
    var_5 = list(var_4)
    var_6 = 'from pandas import DataFrame as df'
    var_7 = [var_3, var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = module_0.imports(var_2)
    var_11 = list(var_10)
    var_12 = 'from os import ('
    var_13 = '    environ'
    var_14 = ')'
    var_15 = [var_12, var_9, var_13, var_14]
    var_16 = module_0.imports(var_15)
    with pytest.raises(AttributeError):
        var_17 = list(var_16)

def test_case_16():
    var_0 = ',\'uSYw\t2*VM\t%~m`\t"}'
    var_1 = 'import sys'
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
    var_5 = 'from os import path'
    var_6 = 'from sys import argv'
    var_7 = [var_5, var_6]
    var_8 = module_0.imports(var_7)
    var_9 = list(var_8)
    var_10 = 'import numpy as np'
    var_11 = 'from pandas import DataFrame as df'
    var_12 = [var_10, var_11]
    var_13 = module_0.imports(var_12)
    var_14 = list(var_13)
    var_15 = 'cimport numpy'
    var_16 = 'from cython cimport int'
    var_17 = [var_15, var_16]
    var_18 = 'cython'
    var_19 = module_0.imports(var_17)
    var_20 = list(var_19)
    var_21 = 'from os import ('
    var_22 = '    path,'
    var_23 = '    environ'
    var_24 = [var_21, var_22, var_23, var_10]
    var_25 = module_0.imports(var_24)
    var_26 = list(var_25)
    var_27 = '    import os'
    var_28 = '        import sys'
    var_29 = [var_27, var_28]
    var_30 = module_0.imports(var_29)
    var_31 = list(var_30)
    var_32 = '# This is a comment'
    var_33 = 'import os  # inline comment'
    var_34 = [var_32, var_33, var_18, var_1]
    var_35 = module_0.imports(var_34)
    var_36 = list(var_35)
    var_37 = 'import os as os'
    var_38 = 'from sys import argv as argv'
    var_39 = [var_37, var_38]
    var_40 = True
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
    var_42 = module_0.imports(var_39, var_41)
    var_43 = list(var_42)
    var_44 = 'def foo():'
    var_45 = '    import sys'
    var_46 = [var_0, var_44, var_45]
    var_47 = True
    var_48 = module_0.imports(var_46, top_only=var_47)
    var_49 = list(var_48)

def test_case_17():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0, var_0]
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
    var_3 = [var_1, var_2, var_0, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    var_5 = var_4.statement()
    assert var_5 == 'from None import :None import None'

def test_case_18():
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
    var_4 = 'import numpy as np'
    var_5 = [var_4]
    var_6 = module_0.imports(var_5)
    var_7 = list(var_6)
    var_8 = 'from sys import argv'
    var_9 = [var_8]
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = 'cimport numpy'
    var_13 = [var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = 'import os, sys'
    var_17 = [var_16]
    var_18 = module_0.imports(var_17)
    var_19 = list(var_18)
    var_20 = '    import os'
    var_21 = [var_20]
    var_22 = module_0.imports(var_21)
    var_23 = list(var_22)
    var_24 = 'from os import (\n    path,\n    sep\n)'
    var_25 = [var_24]
    var_26 = module_0.imports(var_25)
    var_27 = list(var_26)
    var_28 = list(var_23)
    var_29 = True
    var_30 = module_3.Config()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'isort.settings.Config'
    assert var_30.py_version == 'py3'
    assert f'{type(var_30.force_to_top).__module__}.{type(var_30.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.force_to_top) == 0
    assert f'{type(var_30.skip).__module__}.{type(var_30.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.skip) == 19
    assert f'{type(var_30.extend_skip).__module__}.{type(var_30.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.extend_skip) == 0
    assert f'{type(var_30.skip_glob).__module__}.{type(var_30.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.skip_glob) == 0
    assert f'{type(var_30.extend_skip_glob).__module__}.{type(var_30.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.extend_skip_glob) == 0
    assert var_30.skip_gitignore is False
    assert var_30.line_length == 79
    assert var_30.wrap_length == 0
    assert var_30.line_ending == ''
    assert var_30.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_30.no_sections is False
    assert f'{type(var_30.known_future_library).__module__}.{type(var_30.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.known_future_library) == 1
    assert f'{type(var_30.known_third_party).__module__}.{type(var_30.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.known_third_party) == 0
    assert f'{type(var_30.known_first_party).__module__}.{type(var_30.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.known_first_party) == 0
    assert f'{type(var_30.known_local_folder).__module__}.{type(var_30.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.known_local_folder) == 0
    assert f'{type(var_30.known_standard_library).__module__}.{type(var_30.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.known_standard_library) == 234
    assert f'{type(var_30.extra_standard_library).__module__}.{type(var_30.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.extra_standard_library) == 0
    assert var_30.known_other == {}
    assert var_30.multi_line_output == module_1.WrapModes.GRID
    assert var_30.forced_separate == ()
    assert var_30.indent == '    '
    assert var_30.comment_prefix == '  #'
    assert var_30.length_sort is False
    assert var_30.length_sort_straight is False
    assert f'{type(var_30.length_sort_sections).__module__}.{type(var_30.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.length_sort_sections) == 0
    assert f'{type(var_30.add_imports).__module__}.{type(var_30.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.add_imports) == 0
    assert f'{type(var_30.remove_imports).__module__}.{type(var_30.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.remove_imports) == 0
    assert var_30.append_only is False
    assert var_30.reverse_relative is False
    assert var_30.force_single_line is False
    assert var_30.single_line_exclusions == ()
    assert var_30.default_section == 'THIRDPARTY'
    assert var_30.import_headings == {}
    assert var_30.import_footers == {}
    assert var_30.balanced_wrapping is False
    assert var_30.use_parentheses is False
    assert var_30.order_by_type is True
    assert var_30.atomic is False
    assert var_30.lines_before_imports == -1
    assert var_30.lines_after_imports == -1
    assert var_30.lines_between_sections == 1
    assert var_30.lines_between_types == 0
    assert var_30.combine_as_imports is False
    assert var_30.combine_star is False
    assert var_30.include_trailing_comma is False
    assert var_30.from_first is False
    assert var_30.verbose is False
    assert var_30.quiet is False
    assert var_30.force_adds is False
    assert var_30.force_alphabetical_sort_within_sections is False
    assert var_30.force_alphabetical_sort is False
    assert var_30.force_grid_wrap == 0
    assert var_30.force_sort_within_sections is False
    assert var_30.lexicographical is False
    assert var_30.group_by_package is False
    assert var_30.ignore_whitespace is False
    assert f'{type(var_30.no_lines_before).__module__}.{type(var_30.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.no_lines_before) == 0
    assert var_30.no_inline_sort is False
    assert var_30.ignore_comments is False
    assert var_30.case_sensitive is False
    assert f'{type(var_30.sources).__module__}.{type(var_30.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_30.sources) == 1
    assert var_30.virtual_env == ''
    assert var_30.conda_env == ''
    assert var_30.ensure_newline_before_comments is False
    assert var_30.directory == '/workspace'
    assert var_30.profile == ''
    assert var_30.honor_noqa is False
    assert f'{type(var_30.src_paths).__module__}.{type(var_30.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_30.src_paths) == 2
    assert var_30.remove_redundant_aliases is False
    assert var_30.float_to_top is False
    assert var_30.filter_files is False
    assert var_30.formatter == ''
    assert var_30.formatting_function is None
    assert var_30.color_output is False
    assert f'{type(var_30.treat_comments_as_code).__module__}.{type(var_30.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.treat_comments_as_code) == 0
    assert var_30.treat_all_comments_as_code is False
    assert f'{type(var_30.supported_extensions).__module__}.{type(var_30.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.supported_extensions) == 4
    assert f'{type(var_30.blocked_extensions).__module__}.{type(var_30.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.blocked_extensions) == 1
    assert f'{type(var_30.constants).__module__}.{type(var_30.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.constants) == 0
    assert f'{type(var_30.classes).__module__}.{type(var_30.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.classes) == 0
    assert f'{type(var_30.variables).__module__}.{type(var_30.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.variables) == 0
    assert var_30.dedup_headings is False
    assert var_30.only_sections is False
    assert var_30.only_modified is False
    assert var_30.combine_straight_imports is False
    assert var_30.auto_identify_namespace_packages is True
    assert f'{type(var_30.namespace_packages).__module__}.{type(var_30.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_30.namespace_packages) == 0
    assert var_30.follow_links is True
    assert var_30.indented_import_headings is True
    assert var_30.honor_case_in_force_sorted_sections is False
    assert var_30.sort_relative_in_force_sorted_sections is False
    assert var_30.overwrite_in_place is False
    assert var_30.reverse_sort is False
    assert var_30.star_first is False
    assert var_30.git_ls_files == {}
    assert var_30.format_error == '{error}: {message}'
    assert var_30.format_success == '{success}: {message}'
    assert var_30.sort_order == 'natural'
    assert var_30.sort_reexports is False
    assert var_30.split_on_trailing_comma is False
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
    var_31 = 'import numpy as numpy'
    var_32 = [var_31]
    var_33 = module_0.imports(var_32, var_30)
    var_34 = list(var_33)
    var_35 = list(var_28)
    var_36 = []
    var_37 = module_0.imports(var_36)
    var_38 = list(var_37)
    var_39 = list(var_2)
    var_40 = 'def foo():'
    var_41 = '    import sys'
    var_42 = [var_0, var_40, var_41]
    var_43 = True
    var_44 = module_0.imports(var_42, top_only=var_43)
    var_45 = list(var_44)

def test_case_19():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0, var_0]
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
    var_3 = [var_1, var_2, var_0, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    var_5 = var_4.__str__()
    assert var_5 == ':[None, None, None, None, None] indented from None import :None import None'

def test_case_20():
    var_0 = 'sys'
    var_1 = 'import numpy as np\n'
    var_2 = 'import pandas as pd\n'
    var_3 = [var_1, var_2]
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
    var_5 = list(var_4)
    var_6 = module_0.imports(var_3)
    var_7 = list(var_6)
    var_8 = len(var_7)
    assert var_8 == 2
    var_9 = 'cimport numpy as np\n'
    var_10 = [var_9, var_0]
    var_11 = module_0.imports(var_10)
    var_12 = len(var_7)
    assert var_12 == 2
    var_13 = 'from os import (\n'
    var_14 = '    ath\n'
    var_15 = '    environ\n'
    var_16 = [var_13, var_14, var_15, var_0]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = len(var_18)
    var_20 = '    import os\n'
    var_21 = '    import sys\n'
    var_22 = [var_20, var_21]
    var_23 = module_0.imports(var_22)
    var_24 = len(var_9)
    var_25 = list(var_13)
    var_26 = True
    var_27 = module_3.Config()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'isort.settings.Config'
    assert var_27.py_version == 'py3'
    assert f'{type(var_27.force_to_top).__module__}.{type(var_27.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.force_to_top) == 0
    assert f'{type(var_27.skip).__module__}.{type(var_27.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip) == 19
    assert f'{type(var_27.extend_skip).__module__}.{type(var_27.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip) == 0
    assert f'{type(var_27.skip_glob).__module__}.{type(var_27.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.skip_glob) == 0
    assert f'{type(var_27.extend_skip_glob).__module__}.{type(var_27.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extend_skip_glob) == 0
    assert var_27.skip_gitignore is False
    assert var_27.line_length == 79
    assert var_27.wrap_length == 0
    assert var_27.line_ending == ''
    assert var_27.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_27.no_sections is False
    assert f'{type(var_27.known_future_library).__module__}.{type(var_27.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_future_library) == 1
    assert f'{type(var_27.known_third_party).__module__}.{type(var_27.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_third_party) == 0
    assert f'{type(var_27.known_first_party).__module__}.{type(var_27.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_first_party) == 0
    assert f'{type(var_27.known_local_folder).__module__}.{type(var_27.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_local_folder) == 0
    assert f'{type(var_27.known_standard_library).__module__}.{type(var_27.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.known_standard_library) == 234
    assert f'{type(var_27.extra_standard_library).__module__}.{type(var_27.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.extra_standard_library) == 0
    assert var_27.known_other == {}
    assert var_27.multi_line_output == module_1.WrapModes.GRID
    assert var_27.forced_separate == ()
    assert var_27.indent == '    '
    assert var_27.comment_prefix == '  #'
    assert var_27.length_sort is False
    assert var_27.length_sort_straight is False
    assert f'{type(var_27.length_sort_sections).__module__}.{type(var_27.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.length_sort_sections) == 0
    assert f'{type(var_27.add_imports).__module__}.{type(var_27.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.add_imports) == 0
    assert f'{type(var_27.remove_imports).__module__}.{type(var_27.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.remove_imports) == 0
    assert var_27.append_only is False
    assert var_27.reverse_relative is False
    assert var_27.force_single_line is False
    assert var_27.single_line_exclusions == ()
    assert var_27.default_section == 'THIRDPARTY'
    assert var_27.import_headings == {}
    assert var_27.import_footers == {}
    assert var_27.balanced_wrapping is False
    assert var_27.use_parentheses is False
    assert var_27.order_by_type is True
    assert var_27.atomic is False
    assert var_27.lines_before_imports == -1
    assert var_27.lines_after_imports == -1
    assert var_27.lines_between_sections == 1
    assert var_27.lines_between_types == 0
    assert var_27.combine_as_imports is False
    assert var_27.combine_star is False
    assert var_27.include_trailing_comma is False
    assert var_27.from_first is False
    assert var_27.verbose is False
    assert var_27.quiet is False
    assert var_27.force_adds is False
    assert var_27.force_alphabetical_sort_within_sections is False
    assert var_27.force_alphabetical_sort is False
    assert var_27.force_grid_wrap == 0
    assert var_27.force_sort_within_sections is False
    assert var_27.lexicographical is False
    assert var_27.group_by_package is False
    assert var_27.ignore_whitespace is False
    assert f'{type(var_27.no_lines_before).__module__}.{type(var_27.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.no_lines_before) == 0
    assert var_27.no_inline_sort is False
    assert var_27.ignore_comments is False
    assert var_27.case_sensitive is False
    assert f'{type(var_27.sources).__module__}.{type(var_27.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_27.sources) == 1
    assert var_27.virtual_env == ''
    assert var_27.conda_env == ''
    assert var_27.ensure_newline_before_comments is False
    assert var_27.directory == '/workspace'
    assert var_27.profile == ''
    assert var_27.honor_noqa is False
    assert f'{type(var_27.src_paths).__module__}.{type(var_27.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_27.src_paths) == 2
    assert var_27.remove_redundant_aliases is False
    assert var_27.float_to_top is False
    assert var_27.filter_files is False
    assert var_27.formatter == ''
    assert var_27.formatting_function is None
    assert var_27.color_output is False
    assert f'{type(var_27.treat_comments_as_code).__module__}.{type(var_27.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.treat_comments_as_code) == 0
    assert var_27.treat_all_comments_as_code is False
    assert f'{type(var_27.supported_extensions).__module__}.{type(var_27.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.supported_extensions) == 4
    assert f'{type(var_27.blocked_extensions).__module__}.{type(var_27.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.blocked_extensions) == 1
    assert f'{type(var_27.constants).__module__}.{type(var_27.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.constants) == 0
    assert f'{type(var_27.classes).__module__}.{type(var_27.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.classes) == 0
    assert f'{type(var_27.variables).__module__}.{type(var_27.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.variables) == 0
    assert var_27.dedup_headings is False
    assert var_27.only_sections is False
    assert var_27.only_modified is False
    assert var_27.combine_straight_imports is False
    assert var_27.auto_identify_namespace_packages is True
    assert f'{type(var_27.namespace_packages).__module__}.{type(var_27.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_27.namespace_packages) == 0
    assert var_27.follow_links is True
    assert var_27.indented_import_headings is True
    assert var_27.honor_case_in_force_sorted_sections is False
    assert var_27.sort_relative_in_force_sorted_sections is False
    assert var_27.overwrite_in_place is False
    assert var_27.reverse_sort is False
    assert var_27.star_first is False
    assert var_27.git_ls_files == {}
    assert var_27.format_error == '{error}: {message}'
    assert var_27.format_success == '{success}: {message}'
    assert var_27.sort_order == 'natural'
    assert var_27.sort_reexports is False
    assert var_27.split_on_trailing_comma is False
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
    var_28 = module_0.imports(var_3, var_27)
    var_29 = list(var_28)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = 'import os  # comment\n'
    var_32 = 'import sys  # another comment\n'
    var_33 = [var_31, var_32]
    var_34 = module_0.imports(var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = 'import os; import sys\n'
    var_38 = [var_37]
    var_39 = module_0.imports(var_38)
    var_40 = list(var_14)
    var_41 = len(var_40)
    var_42 = 'import os \\\n'
    var_43 = [var_42, var_20]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    var_47 = module_0.imports(var_46)
    with pytest.raises(TypeError):
        var_48 = list(var_47)

def test_case_21():
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
    var_6 = 'sys'
    var_7 = 'from os import path\n'
    var_8 = 'from sys import argv\n'
    var_9 = [var_7, var_8]
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = len(var_11)
    assert var_12 == 2
    var_13 = 'argv'
    var_14 = 'import numpy as np\n'
    var_15 = 'import pandas as pd\n'
    var_16 = [var_14, var_15]
    var_17 = module_0.imports(var_16)
    var_18 = list(var_17)
    var_19 = 'np'
    var_20 = 'from os import path as p\n'
    var_21 = [var_20, var_13]
    var_22 = module_0.imports(var_21)
    var_23 = len(var_19)
    assert var_23 == 2
    var_24 = 'cimport numpy as np\n'
    var_25 = [var_24, var_13]
    var_26 = module_0.imports(var_25)
    var_27 = list(var_26)
    var_28 = len(var_27)
    assert var_28 == 2
    var_29 = 'from os import (\n'
    var_30 = '    path,\n'
    var_31 = '    environ\n'
    var_32 = ')\n'
    var_33 = [var_29, var_30, var_31, var_32]
    var_34 = module_0.imports(var_33)
    var_35 = list(var_34)
    var_36 = len(var_35)
    assert var_36 == 2
    var_37 = '    import os\n'
    var_38 = '    import sys\n'
    var_39 = [var_37, var_38]
    var_40 = module_0.imports(var_39)
    var_41 = list(var_40)
    var_42 = len(var_41)
    assert var_42 == 2
    var_43 = len(var_16)
    assert var_43 == 2
    var_44 = [var_0, var_38, var_38]
    var_45 = True
    var_46 = module_0.imports(var_44, top_only=var_45)
    var_47 = len(var_8)
    assert var_47 == 1
    var_48 = True
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
    var_50 = 'import os as os\n'
    var_51 = 'from sys import path as path\n'
    var_52 = [var_50, var_51]
    var_53 = module_0.imports(var_52, var_49)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = list(var_3)
    var_57 = len(var_56)
    assert var_57 == 2
    var_58 = 'import os; import sys\n'
    var_59 = [var_58]
    var_60 = module_0.imports(var_59)
    var_61 = list(var_60)
    var_62 = len(var_61)
    assert var_62 == 2
    var_63 = 'import os \\\n'
    var_64 = [var_63, var_6]
    var_65 = module_0.imports(var_64)
    var_66 = list(var_65)
    var_67 = 'from os import (path, environ)\n'
    var_68 = [var_67]
    var_69 = module_0.imports(var_68)
    with pytest.raises(TypeError):
        var_70 = len(var_48)
    assert var_70 == 2

def test_case_22():
    var_0 = 'sys'
    var_1 = 'from os import path\n'
    var_2 = 'import numpy as np\n'
    var_3 = 'import pandas as pd\n'
    var_4 = [var_2, var_3]
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
    var_7 = 'np'
    var_8 = 'from sys import argv as a\n'
    var_9 = module_0.imports(var_4)
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = 'cimport numpy as np\n'
    var_13 = [var_12, var_0]
    var_14 = module_0.imports(var_13)
    var_15 = len(var_10)
    assert var_15 == 2
    var_16 = 'from os import (\n'
    var_17 = '    ath\n'
    var_18 = '    environ\n'
    var_19 = [var_16, var_17, var_18, var_0]
    var_20 = module_0.imports(var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    var_23 = '    import os\n'
    var_24 = '    import sys\n'
    var_25 = [var_23, var_24]
    var_26 = module_0.imports(var_25)
    var_27 = len(var_1)
    var_28 = len(var_12)
    var_29 = 'def func():\n'
    var_30 = [var_8, var_29, var_24]
    var_31 = True
    var_32 = module_0.imports(var_30, top_only=var_31)
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
    var_35 = 'import os as os\n'
    var_36 = 'from sys import path as path\n'
    var_37 = [var_35, var_36]
    var_38 = module_0.imports(var_37, var_34)
    var_39 = list(var_38)
    var_40 = len(var_39)
    assert var_40 == 2
    var_41 = 'import os  # comment\n'
    var_42 = 'import sys  # another comment\n'
    var_43 = [var_41, var_42]
    var_44 = module_0.imports(var_43)
    var_45 = list(var_44)
    var_46 = len(var_45)
    assert var_46 == 2
    var_47 = 'import os; import sys\n'
    var_48 = [var_47]
    var_49 = module_0.imports(var_48)
    var_50 = list(var_49)
    var_51 = len(var_50)
    assert var_51 == 2
    var_52 = 'import os \\\n'
    var_53 = [var_52, var_23]
    var_54 = module_0.imports(var_53)
    var_55 = list(var_54)
    var_56 = len(var_55)
    var_57 = []
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = len(var_59)
    var_61 = 'yield\n'
    var_62 = [var_61, var_17]
    var_63 = module_0.imports(var_62)
    var_64 = list(var_63)
    var_65 = 'raise\n'
    var_66 = [var_65, var_7]
    var_67 = module_0.imports(var_66)

def test_case_23():
    var_0 = 'sys'
    var_1 = 'from os import path\n'
    var_2 = 'from sys import argv\n'
    var_3 = [var_1, var_2]
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
    var_5 = list(var_4)
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 'import numpy as np\n'
    var_8 = 'import pandas as pd\n'
    var_9 = [var_7, var_8]
    var_10 = module_0.imports(var_9)
    var_11 = list(var_10)
    var_12 = 'np'
    var_13 = 'from os import path as p\n'
    var_14 = 'from sys import argv as a\n'
    var_15 = [var_13, var_14]
    var_16 = module_0.imports(var_15)
    var_17 = list(var_16)
    var_18 = 'cimport numpy as np\n'
    var_19 = 'c3mport`pandas as pd\n'
    var_20 = [var_18, var_19]
    var_21 = module_0.imports(var_20)
    var_22 = len(var_17)
    assert var_22 == 2
    var_23 = 'from os import (\n'
    var_24 = '    ath\n'
    var_25 = '    environ\n'
    var_26 = [var_23, var_24, var_25, var_0]
    var_27 = module_0.imports(var_26)
    var_28 = list(var_27)
    var_29 = len(var_28)
    var_30 = '    import os\n'
    var_31 = '    import sys\n'
    var_32 = [var_30, var_31]
    var_33 = module_0.imports(var_32)
    var_34 = list(var_33)
    var_35 = len(var_34)
    assert var_35 == 2
    var_36 = 'def func():\n'
    var_37 = [var_14, var_36, var_31]
    var_38 = True
    var_39 = module_0.imports(var_37, top_only=var_38)
    var_40 = list(var_23)
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
    var_42 = 'import os as os\n'
    var_43 = 'from sys import path as path\n'
    var_44 = [var_42, var_43]
    var_45 = module_0.imports(var_44, var_41)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = 'import os  # comment\n'
    var_49 = list(var_48)
    var_50 = len(var_49)
    var_51 = 'import os; import sys\n'
    var_52 = [var_51]
    var_53 = module_0.imports(var_52)
    var_54 = list(var_53)
    var_55 = len(var_54)
    assert var_55 == 2
    var_56 = 'import os \\\n'
    var_57 = [var_56, var_30]
    var_58 = module_0.imports(var_57)
    var_59 = list(var_58)
    var_60 = list(var_27)
    var_61 = len(var_60)
    var_62 = 'from os cimport path\n'
    var_63 = [var_62]
    var_64 = module_0.imports(var_63)
    var_65 = 'yield\n'
    var_66 = [var_65, var_24]
    var_67 = module_0.imports(var_66)
    var_68 = list(var_67)
    var_69 = 'raise\n'
    var_70 = [var_69, var_12]
    var_71 = module_0.imports(var_70)
    var_72 = list(var_71)

def test_case_24():
    var_0 = None
    var_1 = [var_0, var_0, var_0, var_0, var_0]
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
    var_3 = [var_1, var_2, var_1, var_2, var_0, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    var_5 = var_4.statement()
    assert var_5 == 'from [None, None, None, None, None] cimport :None import None'

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'sys'
    var_1 = 'from sys import argv\n'
    var_2 = [var_1, var_1]
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
    var_7 = 'import pandas as pd\n'
    var_8 = [var_6, var_7]
    var_9 = module_0.imports(var_8)
    var_10 = list(var_9)
    var_11 = 'from os import path as p\n'
    var_12 = 'from sys import argv as a\n'
    var_13 = [var_11, var_12]
    var_14 = module_0.imports(var_13)
    var_15 = list(var_14)
    var_16 = 'cimport numpy as np\n'
    var_17 = 'c3mport`pandas as pd\n'
    var_18 = [var_16, var_17]
    var_19 = module_0.imports(var_18)
    var_20 = len(var_15)
    assert var_20 == 2
    var_21 = 'from os import (\n'
    var_22 = '    ath\n'
    var_23 = '    environ\n'
    var_24 = [var_21, var_22, var_23, var_0]
    var_25 = module_0.imports(var_24)
    var_26 = list(var_25)
    var_27 = len(var_26)
    var_28 = '    import sys\n'
    var_29 = list(var_10)
    var_30 = len(var_29)
    assert var_30 == 2
    var_31 = len(var_29)
    assert var_31 == 2
    var_32 = 'def func():\n'
    var_33 = [var_12, var_32, var_28]
    var_34 = True
    var_35 = module_0.imports(var_33, top_only=var_34)
    var_36 = list(var_21)
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
    var_38 = 'import os as os\n'
    var_39 = 'from sys import path as path\n'
    var_40 = [var_38, var_39]
    var_41 = module_0.imports(var_40, var_37)
    var_42 = list(var_41)
    var_43 = len(var_42)
    assert var_43 == 2
    var_44 = 'import os  # comment\n'
    var_45 = 'import sys  # another comment\n'
    var_46 = [var_44, var_45]
    var_47 = module_0.imports(var_46)
    var_48 = list(var_47)
    var_49 = len(var_48)
    assert var_49 == 2
    var_50 = 'import os; import sys\n'
    var_51 = [var_50]
    var_52 = module_0.imports(var_51)
    var_53 = list(var_52)
    var_54 = len(var_53)
    assert var_54 == 2
    var_55 = 'import os \\\n'
    var_56 = [var_55, var_21]
    var_57 = module_0.imports(var_56)
    var_58 = list(var_57)
    var_59 = 'from os import (path, nvron)\n'
    var_60 = [var_59]
    var_61 = module_0.imports(var_60)
    var_62 = list(var_61)
    var_63 = 'from os cimport path\n'
    var_64 = [var_63]
    var_65 = module_0.imports(var_64)
    var_4.__setattr__(var_4, var_15)