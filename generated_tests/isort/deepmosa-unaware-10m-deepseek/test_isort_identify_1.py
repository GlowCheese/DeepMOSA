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

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = var_0.__str__()
    assert var_1 == 'None'
    var_2 = None
    var_3 = module_0.imports(var_1, file_path=var_2, top_only=var_1)
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
    var_4 = [var_2, var_1, var_0, var_1]
    var_5 = module_0.Import(*var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.identify.Import'
    assert len(var_5) == 7
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_6 = var_5.__str__()
    assert var_6 == ':None indented from None import None'
    module_0.Import(*var_3)

def test_case_4():
    var_0 = None
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
    var_2 = None
    var_3 = [var_2, var_2, var_2, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_5 = None
    var_6 = [var_4, var_4, var_5, var_2, var_4]
    var_7 = module_0.Import(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.identify.Import'
    assert len(var_7) == 7
    var_8 = var_7.__str__()
    assert var_8 == '::None import None indented import None as :None import None'

def test_case_5():
    var_0 = None
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
    var_2 = None
    var_3 = [var_2, var_2, var_2, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_5 = None
    var_6 = [var_4, var_4, var_5, var_2, var_3, var_2, var_4]
    var_7 = module_0.Import(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.identify.Import'
    assert len(var_7) == 7
    var_8 = var_7.__str__()
    assert var_8 == ':None import None::None import None indented import None as [None, None, None, None]'

def test_case_6():
    var_0 = None
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
    var_2 = None
    var_3 = [var_2, var_2, var_2, var_2]
    var_4 = module_0.Import(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.identify.Import'
    assert len(var_4) == 7
    assert f'{type(module_0.Import.line_number).__module__}.{type(module_0.Import.line_number).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.indented).__module__}.{type(module_0.Import.indented).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.module).__module__}.{type(module_0.Import.module).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.attribute).__module__}.{type(module_0.Import.attribute).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.alias).__module__}.{type(module_0.Import.alias).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.cimport).__module__}.{type(module_0.Import.cimport).__qualname__}' == '_collections._tuplegetter'
    assert f'{type(module_0.Import.file_path).__module__}.{type(module_0.Import.file_path).__qualname__}' == '_collections._tuplegetter'
    var_5 = None
    var_6 = [var_4, var_2, var_4, var_5, var_2, var_4, var_2]
    var_7 = module_0.Import(*var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'isort.identify.Import'
    assert len(var_7) == 7
    var_8 = var_7.__str__()
    assert var_8 == '::None import None cimport :None import None'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = var_0.__str__()
    assert var_1 == 'None'
    var_2 = module_0.imports(var_1, file_path=var_0)
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
    module_0.Import(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = var_0.__str__()
    assert var_1 == 'None'
    var_2 = module_0.imports(var_1, file_path=var_0, top_only=var_1)
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
    module_0.Import(*var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_2.Config()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'isort.settings.Config'
    assert var_0.py_version == 'py3'
    assert f'{type(var_0.force_to_top).__module__}.{type(var_0.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.force_to_top) == 0
    assert f'{type(var_0.skip).__module__}.{type(var_0.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip) == 19
    assert f'{type(var_0.extend_skip).__module__}.{type(var_0.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip) == 0
    assert f'{type(var_0.skip_glob).__module__}.{type(var_0.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.skip_glob) == 0
    assert f'{type(var_0.extend_skip_glob).__module__}.{type(var_0.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extend_skip_glob) == 0
    assert var_0.skip_gitignore is False
    assert var_0.line_length == 79
    assert var_0.wrap_length == 0
    assert var_0.line_ending == ''
    assert var_0.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_0.no_sections is False
    assert f'{type(var_0.known_future_library).__module__}.{type(var_0.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_future_library) == 1
    assert f'{type(var_0.known_third_party).__module__}.{type(var_0.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_third_party) == 0
    assert f'{type(var_0.known_first_party).__module__}.{type(var_0.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_first_party) == 0
    assert f'{type(var_0.known_local_folder).__module__}.{type(var_0.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_local_folder) == 0
    assert f'{type(var_0.known_standard_library).__module__}.{type(var_0.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.known_standard_library) == 234
    assert f'{type(var_0.extra_standard_library).__module__}.{type(var_0.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.extra_standard_library) == 0
    assert var_0.known_other == {}
    assert var_0.multi_line_output == module_1.WrapModes.GRID
    assert var_0.forced_separate == ()
    assert var_0.indent == '    '
    assert var_0.comment_prefix == '  #'
    assert var_0.length_sort is False
    assert var_0.length_sort_straight is False
    assert f'{type(var_0.length_sort_sections).__module__}.{type(var_0.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.length_sort_sections) == 0
    assert f'{type(var_0.add_imports).__module__}.{type(var_0.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.add_imports) == 0
    assert f'{type(var_0.remove_imports).__module__}.{type(var_0.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.remove_imports) == 0
    assert var_0.append_only is False
    assert var_0.reverse_relative is False
    assert var_0.force_single_line is False
    assert var_0.single_line_exclusions == ()
    assert var_0.default_section == 'THIRDPARTY'
    assert var_0.import_headings == {}
    assert var_0.import_footers == {}
    assert var_0.balanced_wrapping is False
    assert var_0.use_parentheses is False
    assert var_0.order_by_type is True
    assert var_0.atomic is False
    assert var_0.lines_before_imports == -1
    assert var_0.lines_after_imports == -1
    assert var_0.lines_between_sections == 1
    assert var_0.lines_between_types == 0
    assert var_0.combine_as_imports is False
    assert var_0.combine_star is False
    assert var_0.include_trailing_comma is False
    assert var_0.from_first is False
    assert var_0.verbose is False
    assert var_0.quiet is False
    assert var_0.force_adds is False
    assert var_0.force_alphabetical_sort_within_sections is False
    assert var_0.force_alphabetical_sort is False
    assert var_0.force_grid_wrap == 0
    assert var_0.force_sort_within_sections is False
    assert var_0.lexicographical is False
    assert var_0.group_by_package is False
    assert var_0.ignore_whitespace is False
    assert f'{type(var_0.no_lines_before).__module__}.{type(var_0.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.no_lines_before) == 0
    assert var_0.no_inline_sort is False
    assert var_0.ignore_comments is False
    assert var_0.case_sensitive is False
    assert f'{type(var_0.sources).__module__}.{type(var_0.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_0.sources) == 1
    assert var_0.virtual_env == ''
    assert var_0.conda_env == ''
    assert var_0.ensure_newline_before_comments is False
    assert var_0.directory == '/workspace'
    assert var_0.profile == ''
    assert var_0.honor_noqa is False
    assert f'{type(var_0.src_paths).__module__}.{type(var_0.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_0.src_paths) == 2
    assert var_0.remove_redundant_aliases is False
    assert var_0.float_to_top is False
    assert var_0.filter_files is False
    assert var_0.formatter == ''
    assert var_0.formatting_function is None
    assert var_0.color_output is False
    assert f'{type(var_0.treat_comments_as_code).__module__}.{type(var_0.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.treat_comments_as_code) == 0
    assert var_0.treat_all_comments_as_code is False
    assert f'{type(var_0.supported_extensions).__module__}.{type(var_0.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.supported_extensions) == 4
    assert f'{type(var_0.blocked_extensions).__module__}.{type(var_0.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.blocked_extensions) == 1
    assert f'{type(var_0.constants).__module__}.{type(var_0.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.constants) == 0
    assert f'{type(var_0.classes).__module__}.{type(var_0.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.classes) == 0
    assert f'{type(var_0.variables).__module__}.{type(var_0.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.variables) == 0
    assert var_0.dedup_headings is False
    assert var_0.only_sections is False
    assert var_0.only_modified is False
    assert var_0.combine_straight_imports is False
    assert var_0.auto_identify_namespace_packages is True
    assert f'{type(var_0.namespace_packages).__module__}.{type(var_0.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_0.namespace_packages) == 0
    assert var_0.follow_links is True
    assert var_0.indented_import_headings is True
    assert var_0.honor_case_in_force_sorted_sections is False
    assert var_0.sort_relative_in_force_sorted_sections is False
    assert var_0.overwrite_in_place is False
    assert var_0.reverse_sort is False
    assert var_0.star_first is False
    assert var_0.git_ls_files == {}
    assert var_0.format_error == '{error}: {message}'
    assert var_0.format_success == '{success}: {message}'
    assert var_0.sort_order == 'natural'
    assert var_0.sort_reexports is False
    assert var_0.split_on_trailing_comma is False
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
    var_1 = {var_0: var_0}
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
    var_3 = var_0.__repr__()
    assert var_3 == "Config(py_version='py3', force_to_top=frozenset(), skip=frozenset({'.pants.d', '.nox', 'venv', '.hg', '.pytype', 'build', '.venv', '.mypy_cache', '.eggs', '.git', 'buck-out', '_build', '.svn', '.bzr', '.direnv', 'dist', 'node_modules', '__pypackages__', '.tox'}), extend_skip=frozenset(), skip_glob=frozenset(), extend_skip_glob=frozenset(), skip_gitignore=False, line_length=79, wrap_length=0, line_ending='', sections=('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), no_sections=False, known_future_library=frozenset({'__future__'}), known_third_party=frozenset(), known_first_party=frozenset(), known_local_folder=frozenset(), known_standard_library=frozenset({'wsgiref', 'subprocess', 'sre_constants', 'poplib', 'smtpd', 'logging', 'filecmp', 'symtable', 'turtle', 'zipimport', 'mmap', 'ensurepip', 'nt', 'traceback', 'cgitb', 'imaplib', 'cgi', 'cmath', 'dbm', 'string', 'zoneinfo', 'shlex', 'opcode', 'lib2to3', 'csv', 'sched', 'posix', 'imp', 'token', 'fractions', 'pickle', 'ipaddress', 'py_compile', 'tomllib', 'zipapp', 'contextvars', 'nturl2path', 'pprint', 'encodings', 'macpath', 'atexit', 'queue', 'dataclasses', 'xml', 'telnetlib', 'ast', 'msilib', 'lzma', 'multiprocessing', 'json', 'socketserver', 'signal', 'mailbox', 'pyclbr', 'difflib', 'pydoc', 'pipes', 'rlcompleter', 'binascii', 'codecs', 'decimal', 'getpass', 'pydoc_data', 'test', 'datetime', '_ast', 'sqlite3', 'zlib', 'spwd', 'numbers', 'code', 'xxsubtype', 'antigravity', 'genericpath', 'zipfile', 'types', 'wave', 'concurrent', 'marshal', 'gzip', 'compileall', 'symbol', 'xx', 'tracemalloc', 'sys', 'timeit', 'stat', 'readline', 'secrets', 'urllib', 'mimetypes', 'webbrowser', 'bdb', 'linecache', 'tokenize', 'sre', 'nntplib', 'bisect', 'argparse', 'itertools', 'ftplib', 'fpectl', 'profile', 'audioop', 'select', 'syslog', 'uuid', 'mailcap', 'tty', 'pickletools', 'cProfile', 'netrc', 'site', 'array', 'faulthandler', 'pwd', 'unittest', 're', 'shelve', 'math', 'resource', 'uu', 'inspect', 'selectors', 'colorsys', 'weakref', 'trace', 'functools', 'heapq', 'asyncore', 'email', 'hmac', 'tempfile', 'ssl', 'warnings', 'formatter', 'copyreg', 'tkinter', 'operator', 'bz2', 'venv', 'asyncio', 'modulefinder', 'typing', 'this', 'unicodedata', 'turtledemo', 'sre_parse', 'getopt', 'binhex', '_thread', 'fileinput', 'parser', 'tabnanny', 'winsound', 'os', 'locale', 'quopri', 'configparser', 'contextlib', 'gc', 'cmd', 'optparse', 'pathlib', 'textwrap', 'errno', 'hashlib', 'glob', 'enum', 'xxlimited_35', 'io', 'collections', '_dummy_thread', 'builtins', 'posixpath', 'ntpath', 'sysconfig', 'keyword', 'msvcrt', 'fcntl', 'statistics', 'xmlrpc', 'doctest', 'dummy_threading', 'grp', 'shutil', 'pdb', 'curses', 'sre_compile', 'dis', 'socket', 'distutils', 'pty', 'xdrlib', 'winreg', 'abc', 'plistlib', 'annotationlib', 'crypt', 'threading', 'tarfile', 'calendar', 'platform', 'http', 'ctypes', 'time', 'importlib', 'runpy', 'html', 'pkgutil', 'termios', 'struct', 'smtplib', 'pyexpat', 'random', 'sunau', 'stringprep', 'sndhdr', 'gettext', 'chunk', 'xxlimited', 'codeop', 'imghdr', 'base64', 'asynchat', 'fnmatch', 'pstats', 'graphlib', 'aifc', 'ossaudiodev', 'nis', 'reprlib', 'idlelib', 'copy'}), extra_standard_library=frozenset(), known_other={}, multi_line_output=<WrapModes.GRID: 0>, forced_separate=(), indent='    ', comment_prefix='  #', length_sort=False, length_sort_straight=False, length_sort_sections=frozenset(), add_imports=frozenset(), remove_imports=frozenset(), append_only=False, reverse_relative=False, force_single_line=False, single_line_exclusions=(), default_section='THIRDPARTY', import_headings={}, import_footers={}, balanced_wrapping=False, use_parentheses=False, order_by_type=True, atomic=False, lines_before_imports=-1, lines_after_imports=-1, lines_between_sections=1, lines_between_types=0, combine_as_imports=False, combine_star=False, include_trailing_comma=False, from_first=False, verbose=False, quiet=False, force_adds=False, force_alphabetical_sort_within_sections=False, force_alphabetical_sort=False, force_grid_wrap=0, force_sort_within_sections=False, lexicographical=False, group_by_package=False, ignore_whitespace=False, no_lines_before=frozenset(), no_inline_sort=False, ignore_comments=False, case_sensitive=False, sources=({'py_version': 'py3', 'force_to_top': frozenset(), 'skip': frozenset({'.pants.d', '.nox', 'venv', '.hg', '.pytype', 'build', '.venv', '.mypy_cache', '.eggs', '.git', 'buck-out', '_build', '.svn', '.bzr', '.direnv', 'dist', 'node_modules', '__pypackages__', '.tox'}), 'extend_skip': frozenset(), 'skip_glob': frozenset(), 'extend_skip_glob': frozenset(), 'skip_gitignore': False, 'line_length': 79, 'wrap_length': 0, 'line_ending': '', 'sections': ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), 'no_sections': False, 'known_future_library': frozenset({'__future__'}), 'known_third_party': frozenset(), 'known_first_party': frozenset(), 'known_local_folder': frozenset(), 'known_standard_library': frozenset({'wsgiref', 'subprocess', 'sre_constants', 'poplib', 'smtpd', 'logging', 'filecmp', 'symtable', 'turtle', 'zipimport', 'mmap', 'ensurepip', 'nt', 'traceback', 'cgitb', 'imaplib', 'cgi', 'cmath', 'dbm', 'string', 'zoneinfo', 'shlex', 'opcode', 'lib2to3', 'csv', 'sched', 'posix', 'imp', 'token', 'fractions', 'pickle', 'ipaddress', 'py_compile', 'tomllib', 'zipapp', 'contextvars', 'nturl2path', 'pprint', 'encodings', 'macpath', 'atexit', 'queue', 'dataclasses', 'xml', 'telnetlib', 'ast', 'msilib', 'lzma', 'multiprocessing', 'json', 'socketserver', 'signal', 'mailbox', 'pyclbr', 'difflib', 'pydoc', 'pipes', 'rlcompleter', 'binascii', 'codecs', 'decimal', 'getpass', 'pydoc_data', 'test', 'datetime', '_ast', 'sqlite3', 'zlib', 'spwd', 'numbers', 'code', 'xxsubtype', 'antigravity', 'genericpath', 'zipfile', 'types', 'wave', 'concurrent', 'marshal', 'gzip', 'compileall', 'symbol', 'xx', 'tracemalloc', 'sys', 'timeit', 'stat', 'readline', 'secrets', 'urllib', 'mimetypes', 'webbrowser', 'bdb', 'linecache', 'tokenize', 'sre', 'nntplib', 'bisect', 'argparse', 'itertools', 'ftplib', 'fpectl', 'profile', 'audioop', 'select', 'syslog', 'uuid', 'mailcap', 'tty', 'pickletools', 'cProfile', 'netrc', 'site', 'array', 'faulthandler', 'pwd', 'unittest', 're', 'shelve', 'math', 'resource', 'uu', 'inspect', 'selectors', 'colorsys', 'weakref', 'trace', 'functools', 'heapq', 'asyncore', 'email', 'hmac', 'tempfile', 'ssl', 'warnings', 'formatter', 'copyreg', 'tkinter', 'operator', 'bz2', 'venv', 'asyncio', 'modulefinder', 'typing', 'this', 'unicodedata', 'turtledemo', 'sre_parse', 'getopt', 'binhex', '_thread', 'fileinput', 'parser', 'tabnanny', 'winsound', 'os', 'locale', 'quopri', 'configparser', 'contextlib', 'gc', 'cmd', 'optparse', 'pathlib', 'textwrap', 'errno', 'hashlib', 'glob', 'enum', 'xxlimited_35', 'io', 'collections', '_dummy_thread', 'builtins', 'posixpath', 'ntpath', 'sysconfig', 'keyword', 'msvcrt', 'fcntl', 'statistics', 'xmlrpc', 'doctest', 'dummy_threading', 'grp', 'shutil', 'pdb', 'curses', 'sre_compile', 'dis', 'socket', 'distutils', 'pty', 'xdrlib', 'winreg', 'abc', 'plistlib', 'annotationlib', 'crypt', 'threading', 'tarfile', 'calendar', 'platform', 'http', 'ctypes', 'time', 'importlib', 'runpy', 'html', 'pkgutil', 'termios', 'struct', 'smtplib', 'pyexpat', 'random', 'sunau', 'stringprep', 'sndhdr', 'gettext', 'chunk', 'xxlimited', 'codeop', 'imghdr', 'base64', 'asynchat', 'fnmatch', 'pstats', 'graphlib', 'aifc', 'ossaudiodev', 'nis', 'reprlib', 'idlelib', 'copy'}), 'extra_standard_library': frozenset(), 'known_other': {}, 'multi_line_output': <WrapModes.GRID: 0>, 'forced_separate': (), 'indent': '    ', 'comment_prefix': '  #', 'length_sort': False, 'length_sort_straight': False, 'length_sort_sections': frozenset(), 'add_imports': frozenset(), 'remove_imports': frozenset(), 'append_only': False, 'reverse_relative': False, 'force_single_line': False, 'single_line_exclusions': (), 'default_section': 'THIRDPARTY', 'import_headings': {}, 'import_footers': {}, 'balanced_wrapping': False, 'use_parentheses': False, 'order_by_type': True, 'atomic': False, 'lines_before_imports': -1, 'lines_after_imports': -1, 'lines_between_sections': 1, 'lines_between_types': 0, 'combine_as_imports': False, 'combine_star': False, 'include_trailing_comma': False, 'from_first': False, 'verbose': False, 'quiet': False, 'force_adds': False, 'force_alphabetical_sort_within_sections': False, 'force_alphabetical_sort': False, 'force_grid_wrap': 0, 'force_sort_within_sections': False, 'lexicographical': False, 'group_by_package': False, 'ignore_whitespace': False, 'no_lines_before': frozenset(), 'no_inline_sort': False, 'ignore_comments': False, 'case_sensitive': False, 'sources': (), 'virtual_env': '', 'conda_env': '', 'ensure_newline_before_comments': False, 'directory': '', 'profile': '', 'honor_noqa': False, 'src_paths': (), 'remove_redundant_aliases': False, 'float_to_top': False, 'filter_files': False, 'formatter': '', 'formatting_function': None, 'color_output': False, 'treat_comments_as_code': frozenset(), 'treat_all_comments_as_code': False, 'supported_extensions': frozenset({'pyx', 'py', 'pyi', 'pxd'}), 'blocked_extensions': frozenset({'pex'}), 'constants': frozenset(), 'classes': frozenset(), 'variables': frozenset(), 'dedup_headings': False, 'only_sections': False, 'only_modified': False, 'combine_straight_imports': False, 'auto_identify_namespace_packages': True, 'namespace_packages': frozenset(), 'follow_links': True, 'indented_import_headings': True, 'honor_case_in_force_sorted_sections': False, 'sort_relative_in_force_sorted_sections': False, 'overwrite_in_place': False, 'reverse_sort': False, 'star_first': False, 'git_ls_files': {}, 'format_error': '{error}: {message}', 'format_success': '{success}: {message}', 'sort_order': 'natural', 'sort_reexports': False, 'split_on_trailing_comma': False, 'source': 'defaults'},), virtual_env='', conda_env='', ensure_newline_before_comments=False, directory='/workspace', profile='', honor_noqa=False, src_paths=(PosixPath('/workspace/src'), PosixPath('/workspace')), remove_redundant_aliases=False, float_to_top=False, filter_files=False, formatter='', formatting_function=None, color_output=False, treat_comments_as_code=frozenset(), treat_all_comments_as_code=False, supported_extensions=frozenset({'pyx', 'py', 'pyi', 'pxd'}), blocked_extensions=frozenset({'pex'}), constants=frozenset(), classes=frozenset(), variables=frozenset(), dedup_headings=False, only_sections=False, only_modified=False, combine_straight_imports=False, auto_identify_namespace_packages=True, namespace_packages=frozenset(), follow_links=True, indented_import_headings=True, honor_case_in_force_sorted_sections=False, sort_relative_in_force_sorted_sections=False, overwrite_in_place=False, reverse_sort=False, star_first=False, git_ls_files={}, format_error='{error}: {message}', format_success='{success}: {message}', sort_order='natural', sort_reexports=False, split_on_trailing_comma=False)"
    var_4 = None
    var_5 = module_0.imports(var_3, file_path=var_4, top_only=var_3)
    module_0.Import(*var_5)