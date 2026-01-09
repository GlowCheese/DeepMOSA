# Check out: https://github.com/GlowCheese/deepmosa
import isort.files as module_0
import isort.settings as module_1
import isort.wrap_modes as module_2
import pytest


def test_case_0():
    var_0 = None
    var_1 = module_0.find(var_0, var_0, var_0, var_0)

def test_case_1():
    var_0 = "o\x0c4QF'lYRKG.."
    var_1 = module_0.find(var_0, var_0, var_0, var_0)
    with pytest.raises(AttributeError):
        var_2 = list(var_1)

def test_case_2():
    var_0 = ':^'
    var_1 = [var_0]
    var_2 = [var_0]
    var_3 = module_0.find(var_2, var_0, var_1, var_1)
    var_4 = list(var_3)

def test_case_3():
    var_0 = 'test_dir'
    var_1 = [var_0]
    var_2 = module_1.Config()
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
    assert var_2.multi_line_output == module_2.WrapModes.GRID
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
    assert module_1.TYPE_CHECKING is False
    assert module_1.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_1.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_1.FIRSTPARTY == 'FIRSTPARTY'
    assert module_1.FUTURE == 'FUTURE'
    assert module_1.LOCALFOLDER == 'LOCALFOLDER'
    assert module_1.STDLIB == 'STDLIB'
    assert module_1.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_1.CYTHON_EXTENSIONS).__module__}.{type(module_1.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_1.SUPPORTED_EXTENSIONS).__module__}.{type(module_1.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_1.BLOCKED_EXTENSIONS).__module__}.{type(module_1.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.BLOCKED_EXTENSIONS) == 1
    assert module_1.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_1.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_1.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_1.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_1.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_1.DEFAULT_SKIP).__module__}.{type(module_1.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_1.DEFAULT_SKIP) == 19
    assert module_1.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_1.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_1.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_1.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_1.KNOWN_PREFIX == 'known_'
    assert module_1.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_1.RUNTIME_SOURCE == 'runtime'
    assert module_1.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
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
    assert module_1.DEFAULT_CONFIG.multi_line_output == module_2.WrapModes.GRID
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
    assert f'{type(module_1.Config.known_patterns).__module__}.{type(module_1.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.section_comments).__module__}.{type(module_1.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.section_comments_end).__module__}.{type(module_1.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.skips).__module__}.{type(module_1.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.skip_globs).__module__}.{type(module_1.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_1.Config.sorting_function).__module__}.{type(module_1.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_3 = []
    var_4 = []
    var_5 = module_0.find(var_1, var_2, var_3, var_4)
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = [var_7]
    var_9 = module_1.Config()
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
    assert var_9.multi_line_output == module_2.WrapModes.GRID
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
    var_10 = []
    var_11 = module_0.find(var_8, var_9, var_6, var_10)
    var_12 = list(var_11)
    var_13 = len(var_12)
    assert var_13 == 0
    var_14 = len(var_8)
    assert var_14 == 1
    var_15 = 'non_existent_path'
    var_16 = [var_15]
    var_17 = module_1.Config()
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
    assert var_17.multi_line_output == module_2.WrapModes.GRID
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
    var_18 = []
    var_19 = []
    var_20 = module_0.find(var_16, var_17, var_18, var_19)
    var_21 = list(var_20)
    var_22 = len(var_21)
    assert var_22 == 0
    var_23 = len(var_19)
    assert var_23 == 1
    var_24 = 'test_file.py'
    var_25 = [var_24]
    var_26 = module_1.Config()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'isort.settings.Config'
    assert var_26.py_version == 'py3'
    assert f'{type(var_26.force_to_top).__module__}.{type(var_26.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.force_to_top) == 0
    assert f'{type(var_26.skip).__module__}.{type(var_26.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.skip) == 19
    assert f'{type(var_26.extend_skip).__module__}.{type(var_26.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.extend_skip) == 0
    assert f'{type(var_26.skip_glob).__module__}.{type(var_26.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.skip_glob) == 0
    assert f'{type(var_26.extend_skip_glob).__module__}.{type(var_26.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.extend_skip_glob) == 0
    assert var_26.skip_gitignore is False
    assert var_26.line_length == 79
    assert var_26.wrap_length == 0
    assert var_26.line_ending == ''
    assert var_26.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_26.no_sections is False
    assert f'{type(var_26.known_future_library).__module__}.{type(var_26.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.known_future_library) == 1
    assert f'{type(var_26.known_third_party).__module__}.{type(var_26.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.known_third_party) == 0
    assert f'{type(var_26.known_first_party).__module__}.{type(var_26.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.known_first_party) == 0
    assert f'{type(var_26.known_local_folder).__module__}.{type(var_26.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.known_local_folder) == 0
    assert f'{type(var_26.known_standard_library).__module__}.{type(var_26.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.known_standard_library) == 234
    assert f'{type(var_26.extra_standard_library).__module__}.{type(var_26.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.extra_standard_library) == 0
    assert var_26.known_other == {}
    assert var_26.multi_line_output == module_2.WrapModes.GRID
    assert var_26.forced_separate == ()
    assert var_26.indent == '    '
    assert var_26.comment_prefix == '  #'
    assert var_26.length_sort is False
    assert var_26.length_sort_straight is False
    assert f'{type(var_26.length_sort_sections).__module__}.{type(var_26.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.length_sort_sections) == 0
    assert f'{type(var_26.add_imports).__module__}.{type(var_26.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.add_imports) == 0
    assert f'{type(var_26.remove_imports).__module__}.{type(var_26.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.remove_imports) == 0
    assert var_26.append_only is False
    assert var_26.reverse_relative is False
    assert var_26.force_single_line is False
    assert var_26.single_line_exclusions == ()
    assert var_26.default_section == 'THIRDPARTY'
    assert var_26.import_headings == {}
    assert var_26.import_footers == {}
    assert var_26.balanced_wrapping is False
    assert var_26.use_parentheses is False
    assert var_26.order_by_type is True
    assert var_26.atomic is False
    assert var_26.lines_before_imports == -1
    assert var_26.lines_after_imports == -1
    assert var_26.lines_between_sections == 1
    assert var_26.lines_between_types == 0
    assert var_26.combine_as_imports is False
    assert var_26.combine_star is False
    assert var_26.include_trailing_comma is False
    assert var_26.from_first is False
    assert var_26.verbose is False
    assert var_26.quiet is False
    assert var_26.force_adds is False
    assert var_26.force_alphabetical_sort_within_sections is False
    assert var_26.force_alphabetical_sort is False
    assert var_26.force_grid_wrap == 0
    assert var_26.force_sort_within_sections is False
    assert var_26.lexicographical is False
    assert var_26.group_by_package is False
    assert var_26.ignore_whitespace is False
    assert f'{type(var_26.no_lines_before).__module__}.{type(var_26.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.no_lines_before) == 0
    assert var_26.no_inline_sort is False
    assert var_26.ignore_comments is False
    assert var_26.case_sensitive is False
    assert f'{type(var_26.sources).__module__}.{type(var_26.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_26.sources) == 1
    assert var_26.virtual_env == ''
    assert var_26.conda_env == ''
    assert var_26.ensure_newline_before_comments is False
    assert var_26.directory == '/workspace'
    assert var_26.profile == ''
    assert var_26.honor_noqa is False
    assert f'{type(var_26.src_paths).__module__}.{type(var_26.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_26.src_paths) == 2
    assert var_26.remove_redundant_aliases is False
    assert var_26.float_to_top is False
    assert var_26.filter_files is False
    assert var_26.formatter == ''
    assert var_26.formatting_function is None
    assert var_26.color_output is False
    assert f'{type(var_26.treat_comments_as_code).__module__}.{type(var_26.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.treat_comments_as_code) == 0
    assert var_26.treat_all_comments_as_code is False
    assert f'{type(var_26.supported_extensions).__module__}.{type(var_26.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.supported_extensions) == 4
    assert f'{type(var_26.blocked_extensions).__module__}.{type(var_26.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.blocked_extensions) == 1
    assert f'{type(var_26.constants).__module__}.{type(var_26.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.constants) == 0
    assert f'{type(var_26.classes).__module__}.{type(var_26.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.classes) == 0
    assert f'{type(var_26.variables).__module__}.{type(var_26.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.variables) == 0
    assert var_26.dedup_headings is False
    assert var_26.only_sections is False
    assert var_26.only_modified is False
    assert var_26.combine_straight_imports is False
    assert var_26.auto_identify_namespace_packages is True
    assert f'{type(var_26.namespace_packages).__module__}.{type(var_26.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_26.namespace_packages) == 0
    assert var_26.follow_links is True
    assert var_26.indented_import_headings is True
    assert var_26.honor_case_in_force_sorted_sections is False
    assert var_26.sort_relative_in_force_sorted_sections is False
    assert var_26.overwrite_in_place is False
    assert var_26.reverse_sort is False
    assert var_26.star_first is False
    assert var_26.git_ls_files == {}
    assert var_26.format_error == '{error}: {message}'
    assert var_26.format_success == '{success}: {message}'
    assert var_26.sort_order == 'natural'
    assert var_26.sort_reexports is False
    assert var_26.split_on_trailing_comma is False
    var_27 = []
    var_28 = []
    var_29 = module_0.find(var_25, var_26, var_27, var_28)
    var_30 = list(var_29)
    var_31 = len(var_30)
    assert var_31 == 1
    var_32 = 'parent_dir'
    var_33 = [var_32]
    var_34 = module_1.Config()
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
    assert var_34.multi_line_output == module_2.WrapModes.GRID
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
    var_35 = []
    var_36 = []
    var_37 = module_0.find(var_33, var_34, var_35, var_36)
    var_38 = list(var_37)
    var_39 = len(var_38)
    assert var_39 == 3
    var_40 = 'parent_dir_with_skipped'
    var_41 = [var_40]
    var_42 = module_1.Config()
    assert f'{type(var_42).__module__}.{type(var_42).__qualname__}' == 'isort.settings.Config'
    assert var_42.py_version == 'py3'
    assert f'{type(var_42.force_to_top).__module__}.{type(var_42.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.force_to_top) == 0
    assert f'{type(var_42.skip).__module__}.{type(var_42.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.skip) == 19
    assert f'{type(var_42.extend_skip).__module__}.{type(var_42.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.extend_skip) == 0
    assert f'{type(var_42.skip_glob).__module__}.{type(var_42.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.skip_glob) == 0
    assert f'{type(var_42.extend_skip_glob).__module__}.{type(var_42.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.extend_skip_glob) == 0
    assert var_42.skip_gitignore is False
    assert var_42.line_length == 79
    assert var_42.wrap_length == 0
    assert var_42.line_ending == ''
    assert var_42.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_42.no_sections is False
    assert f'{type(var_42.known_future_library).__module__}.{type(var_42.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.known_future_library) == 1
    assert f'{type(var_42.known_third_party).__module__}.{type(var_42.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.known_third_party) == 0
    assert f'{type(var_42.known_first_party).__module__}.{type(var_42.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.known_first_party) == 0
    assert f'{type(var_42.known_local_folder).__module__}.{type(var_42.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.known_local_folder) == 0
    assert f'{type(var_42.known_standard_library).__module__}.{type(var_42.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.known_standard_library) == 234
    assert f'{type(var_42.extra_standard_library).__module__}.{type(var_42.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.extra_standard_library) == 0
    assert var_42.known_other == {}
    assert var_42.multi_line_output == module_2.WrapModes.GRID
    assert var_42.forced_separate == ()
    assert var_42.indent == '    '
    assert var_42.comment_prefix == '  #'
    assert var_42.length_sort is False
    assert var_42.length_sort_straight is False
    assert f'{type(var_42.length_sort_sections).__module__}.{type(var_42.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.length_sort_sections) == 0
    assert f'{type(var_42.add_imports).__module__}.{type(var_42.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.add_imports) == 0
    assert f'{type(var_42.remove_imports).__module__}.{type(var_42.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.remove_imports) == 0
    assert var_42.append_only is False
    assert var_42.reverse_relative is False
    assert var_42.force_single_line is False
    assert var_42.single_line_exclusions == ()
    assert var_42.default_section == 'THIRDPARTY'
    assert var_42.import_headings == {}
    assert var_42.import_footers == {}
    assert var_42.balanced_wrapping is False
    assert var_42.use_parentheses is False
    assert var_42.order_by_type is True
    assert var_42.atomic is False
    assert var_42.lines_before_imports == -1
    assert var_42.lines_after_imports == -1
    assert var_42.lines_between_sections == 1
    assert var_42.lines_between_types == 0
    assert var_42.combine_as_imports is False
    assert var_42.combine_star is False
    assert var_42.include_trailing_comma is False
    assert var_42.from_first is False
    assert var_42.verbose is False
    assert var_42.quiet is False
    assert var_42.force_adds is False
    assert var_42.force_alphabetical_sort_within_sections is False
    assert var_42.force_alphabetical_sort is False
    assert var_42.force_grid_wrap == 0
    assert var_42.force_sort_within_sections is False
    assert var_42.lexicographical is False
    assert var_42.group_by_package is False
    assert var_42.ignore_whitespace is False
    assert f'{type(var_42.no_lines_before).__module__}.{type(var_42.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.no_lines_before) == 0
    assert var_42.no_inline_sort is False
    assert var_42.ignore_comments is False
    assert var_42.case_sensitive is False
    assert f'{type(var_42.sources).__module__}.{type(var_42.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_42.sources) == 1
    assert var_42.virtual_env == ''
    assert var_42.conda_env == ''
    assert var_42.ensure_newline_before_comments is False
    assert var_42.directory == '/workspace'
    assert var_42.profile == ''
    assert var_42.honor_noqa is False
    assert f'{type(var_42.src_paths).__module__}.{type(var_42.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_42.src_paths) == 2
    assert var_42.remove_redundant_aliases is False
    assert var_42.float_to_top is False
    assert var_42.filter_files is False
    assert var_42.formatter == ''
    assert var_42.formatting_function is None
    assert var_42.color_output is False
    assert f'{type(var_42.treat_comments_as_code).__module__}.{type(var_42.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.treat_comments_as_code) == 0
    assert var_42.treat_all_comments_as_code is False
    assert f'{type(var_42.supported_extensions).__module__}.{type(var_42.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.supported_extensions) == 4
    assert f'{type(var_42.blocked_extensions).__module__}.{type(var_42.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.blocked_extensions) == 1
    assert f'{type(var_42.constants).__module__}.{type(var_42.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.constants) == 0
    assert f'{type(var_42.classes).__module__}.{type(var_42.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.classes) == 0
    assert f'{type(var_42.variables).__module__}.{type(var_42.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.variables) == 0
    assert var_42.dedup_headings is False
    assert var_42.only_sections is False
    assert var_42.only_modified is False
    assert var_42.combine_straight_imports is False
    assert var_42.auto_identify_namespace_packages is True
    assert f'{type(var_42.namespace_packages).__module__}.{type(var_42.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_42.namespace_packages) == 0
    assert var_42.follow_links is True
    assert var_42.indented_import_headings is True
    assert var_42.honor_case_in_force_sorted_sections is False
    assert var_42.sort_relative_in_force_sorted_sections is False
    assert var_42.overwrite_in_place is False
    assert var_42.reverse_sort is False
    assert var_42.star_first is False
    assert var_42.git_ls_files == {}
    assert var_42.format_error == '{error}: {message}'
    assert var_42.format_success == '{success}: {message}'
    assert var_42.sort_order == 'natural'
    assert var_42.sort_reexports is False
    assert var_42.split_on_trailing_comma is False
    var_43 = []
    var_44 = []
    var_45 = module_0.find(var_41, var_42, var_43, var_44)
    var_46 = list(var_45)
    var_47 = len(var_46)
    assert var_47 == 2
    var_48 = len(var_43)
    assert var_48 == 1
    var_49 = 'dir_with_broken_symlinks'
    var_50 = [var_49]
    var_51 = module_1.Config()
    assert f'{type(var_51).__module__}.{type(var_51).__qualname__}' == 'isort.settings.Config'
    assert var_51.py_version == 'py3'
    assert f'{type(var_51.force_to_top).__module__}.{type(var_51.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.force_to_top) == 0
    assert f'{type(var_51.skip).__module__}.{type(var_51.skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.skip) == 19
    assert f'{type(var_51.extend_skip).__module__}.{type(var_51.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.extend_skip) == 0
    assert f'{type(var_51.skip_glob).__module__}.{type(var_51.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.skip_glob) == 0
    assert f'{type(var_51.extend_skip_glob).__module__}.{type(var_51.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.extend_skip_glob) == 0
    assert var_51.skip_gitignore is False
    assert var_51.line_length == 79
    assert var_51.wrap_length == 0
    assert var_51.line_ending == ''
    assert var_51.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert var_51.no_sections is False
    assert f'{type(var_51.known_future_library).__module__}.{type(var_51.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.known_future_library) == 1
    assert f'{type(var_51.known_third_party).__module__}.{type(var_51.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.known_third_party) == 0
    assert f'{type(var_51.known_first_party).__module__}.{type(var_51.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.known_first_party) == 0
    assert f'{type(var_51.known_local_folder).__module__}.{type(var_51.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.known_local_folder) == 0
    assert f'{type(var_51.known_standard_library).__module__}.{type(var_51.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.known_standard_library) == 234
    assert f'{type(var_51.extra_standard_library).__module__}.{type(var_51.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.extra_standard_library) == 0
    assert var_51.known_other == {}
    assert var_51.multi_line_output == module_2.WrapModes.GRID
    assert var_51.forced_separate == ()
    assert var_51.indent == '    '
    assert var_51.comment_prefix == '  #'
    assert var_51.length_sort is False
    assert var_51.length_sort_straight is False
    assert f'{type(var_51.length_sort_sections).__module__}.{type(var_51.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.length_sort_sections) == 0
    assert f'{type(var_51.add_imports).__module__}.{type(var_51.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.add_imports) == 0
    assert f'{type(var_51.remove_imports).__module__}.{type(var_51.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.remove_imports) == 0
    assert var_51.append_only is False
    assert var_51.reverse_relative is False
    assert var_51.force_single_line is False
    assert var_51.single_line_exclusions == ()
    assert var_51.default_section == 'THIRDPARTY'
    assert var_51.import_headings == {}
    assert var_51.import_footers == {}
    assert var_51.balanced_wrapping is False
    assert var_51.use_parentheses is False
    assert var_51.order_by_type is True
    assert var_51.atomic is False
    assert var_51.lines_before_imports == -1
    assert var_51.lines_after_imports == -1
    assert var_51.lines_between_sections == 1
    assert var_51.lines_between_types == 0
    assert var_51.combine_as_imports is False
    assert var_51.combine_star is False
    assert var_51.include_trailing_comma is False
    assert var_51.from_first is False
    assert var_51.verbose is False
    assert var_51.quiet is False
    assert var_51.force_adds is False
    assert var_51.force_alphabetical_sort_within_sections is False
    assert var_51.force_alphabetical_sort is False
    assert var_51.force_grid_wrap == 0
    assert var_51.force_sort_within_sections is False
    assert var_51.lexicographical is False
    assert var_51.group_by_package is False
    assert var_51.ignore_whitespace is False
    assert f'{type(var_51.no_lines_before).__module__}.{type(var_51.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.no_lines_before) == 0
    assert var_51.no_inline_sort is False
    assert var_51.ignore_comments is False
    assert var_51.case_sensitive is False
    assert f'{type(var_51.sources).__module__}.{type(var_51.sources).__qualname__}' == 'builtins.tuple'
    assert len(var_51.sources) == 1
    assert var_51.virtual_env == ''
    assert var_51.conda_env == ''
    assert var_51.ensure_newline_before_comments is False
    assert var_51.directory == '/workspace'
    assert var_51.profile == ''
    assert var_51.honor_noqa is False
    assert f'{type(var_51.src_paths).__module__}.{type(var_51.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(var_51.src_paths) == 2
    assert var_51.remove_redundant_aliases is False
    assert var_51.float_to_top is False
    assert var_51.filter_files is False
    assert var_51.formatter == ''
    assert var_51.formatting_function is None
    assert var_51.color_output is False
    assert f'{type(var_51.treat_comments_as_code).__module__}.{type(var_51.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.treat_comments_as_code) == 0
    assert var_51.treat_all_comments_as_code is False
    assert f'{type(var_51.supported_extensions).__module__}.{type(var_51.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.supported_extensions) == 4
    assert f'{type(var_51.blocked_extensions).__module__}.{type(var_51.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.blocked_extensions) == 1
    assert f'{type(var_51.constants).__module__}.{type(var_51.constants).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.constants) == 0
    assert f'{type(var_51.classes).__module__}.{type(var_51.classes).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.classes) == 0
    assert f'{type(var_51.variables).__module__}.{type(var_51.variables).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.variables) == 0
    assert var_51.dedup_headings is False
    assert var_51.only_sections is False
    assert var_51.only_modified is False
    assert var_51.combine_straight_imports is False
    assert var_51.auto_identify_namespace_packages is True
    assert f'{type(var_51.namespace_packages).__module__}.{type(var_51.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(var_51.namespace_packages) == 0
    assert var_51.follow_links is True
    assert var_51.indented_import_headings is True
    assert var_51.honor_case_in_force_sorted_sections is False
    assert var_51.sort_relative_in_force_sorted_sections is False
    assert var_51.overwrite_in_place is False
    assert var_51.reverse_sort is False
    assert var_51.star_first is False
    assert var_51.git_ls_files == {}
    assert var_51.format_error == '{error}: {message}'
    assert var_51.format_success == '{success}: {message}'
    assert var_51.sort_order == 'natural'
    assert var_51.sort_reexports is False
    assert var_51.split_on_trailing_comma is False
    var_52 = []
    var_53 = module_0.find(var_50, var_51, var_52, var_41)
    var_54 = len(var_3)
    assert var_54 == 0
    with pytest.raises(TypeError):
        var_55 = len(var_22)
    assert var_55 == 1