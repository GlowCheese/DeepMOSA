# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.place as module_0
import isort.wrap_modes as module_1
import pathlib as module_2
import isort.settings as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.module(var_0)

def test_case_1():
    var_0 = '<1'
    var_1 = module_0.module(var_0)
    assert var_1 == 'THIRDPARTY'
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
    assert module_0.LOCAL == 'LOCALFOLDER'

def test_case_2():
    var_0 = 'test_ns_empty'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = 'py'
    var_7 = [var_6]
    var_8 = frozenset(var_7)
    var_9 = module_0._is_namespace_package(var_3, var_8)
    assert var_9 is True
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
    assert module_0.LOCAL == 'LOCALFOLDER'

def test_case_3():
    var_0 = 'a]Da.FBJ6NkC,\rS1kqW'
    var_1 = module_0.module(var_0)
    assert var_1 == 'THIRDPARTY'
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
    assert module_0.LOCAL == 'LOCALFOLDER'

def test_case_4():
    pass

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.module(var_0, var_0)

def test_case_6():
    var_0 = 'test_ns_setup_cfg'
    var_1 = [var_0]
    var_2 = module_0.module(var_0)
    assert var_2 == 'FIRSTPARTY'
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_3 = {}
    var_4 = module_2.Path(*var_1, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_5 = True
    var_6 = var_4.mkdir(exist_ok=var_5)
    var_7 = 'setup.cfg'
    var_8 = var_4 / var_7
    var_9 = 'py'
    var_10 = [var_9]
    var_11 = frozenset(var_10)
    var_12 = module_0._is_namespace_package(var_4, var_11)
    assert var_12 is True

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = '4+'
    var_1 = module_0.module(var_0)
    assert var_1 == 'THIRDPARTY'
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_2 = None
    var_3 = '.e'
    var_4 = module_0.module(var_3)
    assert var_4 == 'LOCALFOLDER'
    var_5 = module_0.module(var_0)
    assert var_5 == 'THIRDPARTY'
    module_0.module(var_2)

def test_case_8():
    var_0 = '/tmp/dummy_src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = [var_3]
    var_5 = set()
    var_6 = False
    var_7 = 'py'
    var_8 = [var_7]
    var_9 = 'src_paths'
    var_10 = 'namespace_packages'
    var_11 = 'auto_identify_namespace_packages'
    var_12 = 'supported_extensions'
    var_13 = {var_9: var_4, var_10: var_5, var_11: var_6, var_12: var_8}
    var_14 = module_3.Config(**var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'isort.settings.Config'
    assert var_14.py_version == 'py3'
    assert f'{type(var_14.force_to_top).__module__}.{type(var_14.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.force_to_top) == 0
    assert f'{type(var_14.skip).__module__}.{type(var_14.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip) == 19
    assert f'{type(var_14.extend_skip).__module__}.{type(var_14.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip) == 0
    assert f'{type(var_14.skip_glob).__module__}.{type(var_14.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.skip_glob) == 0
    assert f'{type(var_14.extend_skip_glob).__module__}.{type(var_14.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extend_skip_glob) == 0
    assert var_14.skip_gitignore is False
    assert var_14.line_length == 79
    assert var_14.wrap_length == 0
    assert var_14.line_ending == ''
    assert var_14.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_14.no_sections is False
    assert f'{type(var_14.known_future_library).__module__}.{type(var_14.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_future_library) == 1
    assert f'{type(var_14.known_third_party).__module__}.{type(var_14.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_third_party) == 0
    assert f'{type(var_14.known_first_party).__module__}.{type(var_14.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_first_party) == 0
    assert f'{type(var_14.known_local_folder).__module__}.{type(var_14.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_local_folder) == 0
    assert f'{type(var_14.known_standard_library).__module__}.{type(var_14.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.known_standard_library) == 234
    assert f'{type(var_14.extra_standard_library).__module__}.{type(var_14.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.extra_standard_library) == 0
    assert var_14.known_other == {}
    assert var_14.multi_line_output == module_1.WrapModes.GRID
    assert var_14.forced_separate == ()
    assert var_14.indent == '    '
    assert var_14.comment_prefix == '  #'
    assert var_14.length_sort is False
    assert var_14.length_sort_straight is False
    assert f'{type(var_14.length_sort_sections).__module__}.{type(var_14.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.length_sort_sections) == 0
    assert f'{type(var_14.add_imports).__module__}.{type(var_14.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.add_imports) == 0
    assert f'{type(var_14.remove_imports).__module__}.{type(var_14.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.remove_imports) == 0
    assert var_14.append_only is False
    assert var_14.reverse_relative is False
    assert var_14.force_single_line is False
    assert var_14.single_line_exclusions == ()
    assert var_14.default_section == 'THIRDPARTY'
    assert var_14.import_headings == {}
    assert var_14.import_footers == {}
    assert var_14.balanced_wrapping is False
    assert var_14.use_parentheses is False
    assert var_14.order_by_type is True
    assert var_14.atomic is False
    assert var_14.lines_before_imports == -1
    assert var_14.lines_after_imports == -1
    assert var_14.lines_between_sections == 1
    assert var_14.lines_between_types == 0
    assert var_14.combine_as_imports is False
    assert var_14.combine_star is False
    assert var_14.include_trailing_comma is False
    assert var_14.from_first is False
    assert var_14.verbose is False
    assert var_14.quiet is False
    assert var_14.force_adds is False
    assert var_14.force_alphabetical_sort_within_sections is False
    assert var_14.force_alphabetical_sort is False
    assert var_14.force_grid_wrap == 0
    assert var_14.force_sort_within_sections is False
    assert var_14.lexicographical is False
    assert var_14.group_by_package is False
    assert var_14.ignore_whitespace is False
    assert f'{type(var_14.no_lines_before).__module__}.{type(var_14.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.no_lines_before) == 0
    assert var_14.no_inline_sort is False
    assert var_14.ignore_comments is False
    assert var_14.case_sensitive is False
    assert f'{type(var_14.sources).__module__}.{type(var_14.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_14.sources) == 2
    assert var_14.virtual_env == ''
    assert var_14.conda_env == ''
    assert var_14.ensure_newline_before_comments is False
    assert var_14.directory == '/workspace/run'
    assert var_14.profile == ''
    assert var_14.honor_noqa is False
    assert f'{type(var_14.src_paths).__module__}.{type(var_14.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_14.src_paths) == 1
    assert var_14.remove_redundant_aliases is False
    assert var_14.float_to_top is False
    assert var_14.filter_files is False
    assert var_14.formatter == ''
    assert var_14.formatting_function is None
    assert var_14.color_output is False
    assert f'{type(var_14.treat_comments_as_code).__module__}.{type(var_14.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.treat_comments_as_code) == 0
    assert var_14.treat_all_comments_as_code is False
    assert f'{type(var_14.supported_extensions).__module__}.{type(var_14.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.supported_extensions) == 1
    assert f'{type(var_14.blocked_extensions).__module__}.{type(var_14.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.blocked_extensions) == 1
    assert f'{type(var_14.constants).__module__}.{type(var_14.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.constants) == 0
    assert f'{type(var_14.classes).__module__}.{type(var_14.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.classes) == 0
    assert f'{type(var_14.variables).__module__}.{type(var_14.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.variables) == 0
    assert var_14.dedup_headings is False
    assert var_14.only_sections is False
    assert var_14.only_modified is False
    assert var_14.combine_straight_imports is False
    assert var_14.auto_identify_namespace_packages is False
    assert f'{type(var_14.namespace_packages).__module__}.{type(var_14.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_14.namespace_packages) == 0
    assert var_14.follow_links is True
    assert var_14.indented_import_headings is True
    assert var_14.honor_case_in_force_sorted_sections is False
    assert var_14.sort_relative_in_force_sorted_sections is False
    assert var_14.overwrite_in_place is False
    assert var_14.reverse_sort is False
    assert var_14.star_first is False
    assert var_14.git_ls_files == {}
    assert var_14.format_error == '{error}: {message}'
    assert var_14.format_success == '{success}: {message}'
    assert var_14.sort_order == 'natural'
    assert var_14.sort_reexports is False
    assert var_14.split_on_trailing_comma is False
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
    assert module_3.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_15 = 'dummy'
    var_16 = [var_0]
    var_17 = {}
    var_18 = module_2.Path(*var_16, **var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pathlib.PosixPath'
    var_19 = [var_18]
    var_20 = module_0._src_path(var_15, var_14, var_19)
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_21 = bool(var_20 is not None or True)
    assert var_21 is True

def test_case_9():
    var_0 = 'abc'
    var_1 = module_0.module(var_0)
    assert var_1 == 'STDLIB'
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
    assert module_0.LOCAL == 'LOCALFOLDER'

def test_case_10():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = [var_3]
    var_5 = 'my_package'
    var_6 = {var_5}
    var_7 = False
    var_8 = '.py'
    var_9 = [var_8]
    var_10 = 'src_paths'
    var_11 = 'namespace_packages'
    var_12 = 'auto_identify_namespace_packages'
    var_13 = 'supported_extensions'
    var_14 = {var_10: var_4, var_11: var_6, var_12: var_7, var_13: var_9}
    var_15 = module_3.Config(**var_14)
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
    assert len(var_15.sources) == 2
    assert var_15.virtual_env == ''
    assert var_15.conda_env == ''
    assert var_15.ensure_newline_before_comments is False
    assert var_15.directory == '/workspace/run'
    assert var_15.profile == ''
    assert var_15.honor_noqa is False
    assert f'{type(var_15.src_paths).__module__}.{type(var_15.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_15.src_paths) == 1
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
    assert len(var_15.supported_extensions) == 1
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
    assert var_15.auto_identify_namespace_packages is False
    assert f'{type(var_15.namespace_packages).__module__}.{type(var_15.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_15.namespace_packages) == 1
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
    assert module_3.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_16 = tuple()
    var_17 = 'my_package.submodule'
    var_18 = [var_0]
    var_19 = {}
    var_20 = module_2.Path(*var_18, **var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'pathlib.PosixPath'
    var_21 = [var_20]
    var_22 = ()
    var_23 = module_0._src_path(var_17, var_15, var_21, var_22)
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_24 = bool(var_23 is not None)

def test_case_11():
    var_0 = 'test_pkg_pkgutil'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = True
    var_5 = var_3.mkdir(exist_ok=var_4)
    var_6 = '__init__.py'
    var_7 = var_3 / var_6
    var_8 = 'S.t<@'
    var_9 = module_0.module(var_8)
    assert var_9 == 'THIRDPARTY'
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_10 = 'py'
    var_11 = [var_10]
    var_12 = frozenset(var_11)
    var_13 = module_0._is_namespace_package(var_3, var_12)
    assert var_13 is True

def test_case_12():
    var_0 = '/tmp/src'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_2.Path(*var_1, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pathlib.PosixPath'
    assert module_2.EINVAL == 22
    assert module_2.ENOENT == 2
    assert module_2.ENOTDIR == 20
    assert module_2.EBADF == 9
    assert module_2.ELOOP == 40
    var_4 = True
    var_5 = var_3.mkdir(parents=var_4, exist_ok=var_4)
    var_6 = [var_3]
    var_7 = 'different.namespace'
    var_8 = {var_7}
    var_9 = False
    var_10 = '.py'
    var_11 = [var_10]
    var_12 = 'src_paths'
    var_13 = 'namespace_packages'
    var_14 = 'auto_identify_namespace_packages'
    var_15 = 'supported_extensions'
    var_16 = {var_12: var_6, var_13: var_8, var_14: var_9, var_15: var_11}
    var_17 = module_3.Config(**var_16)
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
    assert len(var_17.sources) == 2
    assert var_17.virtual_env == ''
    assert var_17.conda_env == ''
    assert var_17.ensure_newline_before_comments is False
    assert var_17.directory == '/workspace/run'
    assert var_17.profile == ''
    assert var_17.honor_noqa is False
    assert f'{type(var_17.src_paths).__module__}.{type(var_17.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_17.src_paths) == 1
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
    assert len(var_17.supported_extensions) == 1
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
    assert var_17.auto_identify_namespace_packages is False
    assert f'{type(var_17.namespace_packages).__module__}.{type(var_17.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_17.namespace_packages) == 1
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
    assert module_3.DEFAULT_CONFIG.directory == '/workspace/run'
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
    var_18 = 'my_module.sub'
    var_19 = ()
    var_20 = module_0._src_path(var_18, var_17, prefix=var_19)
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
    assert module_0.LOCAL == 'LOCALFOLDER'
    var_21 = bool(var_20 is not None)