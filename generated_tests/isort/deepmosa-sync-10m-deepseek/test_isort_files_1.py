# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.files as module_0
import isort.settings as module_1
import isort.wrap_modes as module_2

def test_case_0():
    var_0 = None
    var_1 = module_0.find(var_0, var_0, var_0, var_0)

def test_case_1():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
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
    assert var_1.multi_line_output == module_2.WrapModes.GRID
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
    var_2 = 'skipped_dir'
    var_3 = []
    var_4 = [var_2]
    var_5 = module_0.find(var_4, var_1, var_4, var_3)
    var_6 = list(var_5)
    var_7 = bool(var_3 == ['skipped_dir'])
    assert var_7 is True

def test_case_2():
    var_0 = {}
    var_1 = module_1.Config(**var_0)
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
    assert var_1.multi_line_output == module_2.WrapModes.GRID
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
    var_2 = []
    var_3 = var_1.__repr__()
    assert var_3 == "Config(py_version='py3', force_to_top=frozenset(), skip=frozenset({'.venv', '__pypackages__', '.pants.d', '.hg', '.pytype', '.direnv', '.mypy_cache', '.eggs', 'node_modules', '.nox', 'buck-out', '_build', 'venv', 'build', '.svn', '.tox', 'dist', '.git', '.bzr'}), extend_skip=frozenset(), skip_glob=frozenset(), extend_skip_glob=frozenset(), skip_gitignore=False, line_length=79, wrap_length=0, line_ending='', sections=('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), no_sections=False, known_future_library=frozenset({'__future__'}), known_third_party=frozenset(), known_first_party=frozenset(), known_local_folder=frozenset(), known_standard_library=frozenset({'curses', 'secrets', 'urllib', 'pstats', 'os', 'rlcompleter', 'opcode', 'site', 'traceback', 'collections', 'genericpath', 'tempfile', 'ipaddress', 'pkgutil', 'cmd', 'ast', 'optparse', 'datetime', 'venv', 'mmap', 'string', 'crypt', 'numbers', 'builtins', 'cmath', 'pty', 'pwd', 'colorsys', 'tarfile', 'gzip', 'dataclasses', 'sre', 'decimal', 'ensurepip', 'zlib', 'pathlib', 'antigravity', 'signal', 'pyclbr', 'glob', 'syslog', 'random', 'mailbox', 'imp', 'binascii', 'aifc', 'codeop', 'sre_parse', 'posix', 'symtable', 'nturl2path', 'array', 'dis', 'smtpd', 'sysconfig', 'grp', 'turtledemo', 'unicodedata', 'wave', 'timeit', 'warnings', 'this', 'subprocess', 'annotationlib', 'socket', 'zipapp', 'compileall', 'pprint', 'sre_constants', 'pdb', 'itertools', 'locale', 'mimetypes', 'codecs', 'macpath', 'msilib', 'netrc', 'nis', 'logging', 'telnetlib', 'modulefinder', 'errno', 'trace', 'ossaudiodev', 'encodings', 'functools', 'spwd', 'zipfile', 'configparser', 'gc', 'runpy', '_dummy_thread', 'pyexpat', 'sys', 'webbrowser', 'sndhdr', 'msvcrt', 'http', 'hmac', 'concurrent', 'plistlib', 'abc', 'code', 'marshal', 'bz2', 'graphlib', 'socketserver', 'sunau', 'bdb', 'copyreg', 'typing', 'xdrlib', 'base64', 'dbm', 'zipimport', 'platform', 'xxsubtype', 'calendar', 'formatter', 'keyword', 'contextvars', 'xx', 'hashlib', 'json', 'tabnanny', 'fileinput', 'sched', 'chunk', 'pickle', 'fpectl', 'contextlib', 'quopri', 'turtle', 'statistics', 'stat', 'winsound', 'token', 'zoneinfo', 'parser', 'select', '_ast', 'xxlimited_35', 'reprlib', 'sqlite3', 'termios', 'pydoc_data', 'getopt', 'copy', 'xml', 'imghdr', 'asyncio', 'queue', 'email', 'tokenize', 'uuid', 'selectors', 'shelve', 'math', 'test', 'inspect', 'atexit', 'profile', 'readline', 'stringprep', 'filecmp', 'pydoc', 'html', 'nntplib', 'poplib', 'threading', 'cProfile', 'ntpath', 'binhex', 'doctest', 'pickletools', 'types', 'audioop', 'ftplib', 'wsgiref', 'posixpath', 'csv', 'dummy_threading', 'heapq', 'time', 'weakref', 'operator', 'ctypes', '_thread', 'lib2to3', 'enum', 'tomllib', 'ssl', 'cgi', 'cgitb', 'mailcap', 'shlex', 'idlelib', 'smtplib', 'pipes', 'importlib', 'textwrap', 'fnmatch', 'struct', 'bisect', 'winreg', 'argparse', 'asynchat', 'shutil', 'tty', 'io', 'xmlrpc', 'nt', 'tkinter', 're', 'lzma', 'asyncore', 'faulthandler', 'resource', 'tracemalloc', 'distutils', 'symbol', 'fcntl', 'getpass', 'py_compile', 'linecache', 'fractions', 'imaplib', 'uu', 'unittest', 'xxlimited', 'multiprocessing', 'sre_compile', 'difflib', 'gettext'}), extra_standard_library=frozenset(), known_other={}, multi_line_output=<WrapModes.GRID: 0>, forced_separate=(), indent='    ', comment_prefix='  #', length_sort=False, length_sort_straight=False, length_sort_sections=frozenset(), add_imports=frozenset(), remove_imports=frozenset(), append_only=False, reverse_relative=False, force_single_line=False, single_line_exclusions=(), default_section='THIRDPARTY', import_headings={}, import_footers={}, balanced_wrapping=False, use_parentheses=False, order_by_type=True, atomic=False, lines_before_imports=-1, lines_after_imports=-1, lines_between_sections=1, lines_between_types=0, combine_as_imports=False, combine_star=False, include_trailing_comma=False, from_first=False, verbose=False, quiet=False, force_adds=False, force_alphabetical_sort_within_sections=False, force_alphabetical_sort=False, force_grid_wrap=0, force_sort_within_sections=False, lexicographical=False, group_by_package=False, ignore_whitespace=False, no_lines_before=frozenset(), no_inline_sort=False, ignore_comments=False, case_sensitive=False, sources=({'py_version': 'py3', 'force_to_top': frozenset(), 'skip': frozenset({'.venv', '__pypackages__', '.pants.d', '.hg', '.pytype', '.direnv', '.mypy_cache', '.eggs', 'node_modules', '.nox', 'buck-out', '_build', 'venv', 'build', '.svn', '.tox', 'dist', '.git', '.bzr'}), 'extend_skip': frozenset(), 'skip_glob': frozenset(), 'extend_skip_glob': frozenset(), 'skip_gitignore': False, 'line_length': 79, 'wrap_length': 0, 'line_ending': '', 'sections': ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER'), 'no_sections': False, 'known_future_library': frozenset({'__future__'}), 'known_third_party': frozenset(), 'known_first_party': frozenset(), 'known_local_folder': frozenset(), 'known_standard_library': frozenset({'curses', 'secrets', 'urllib', 'pstats', 'os', 'rlcompleter', 'opcode', 'site', 'traceback', 'collections', 'genericpath', 'tempfile', 'ipaddress', 'pkgutil', 'cmd', 'ast', 'optparse', 'datetime', 'venv', 'mmap', 'string', 'crypt', 'numbers', 'builtins', 'cmath', 'pty', 'pwd', 'colorsys', 'tarfile', 'gzip', 'dataclasses', 'sre', 'decimal', 'ensurepip', 'zlib', 'pathlib', 'antigravity', 'signal', 'pyclbr', 'glob', 'syslog', 'random', 'mailbox', 'imp', 'binascii', 'aifc', 'codeop', 'sre_parse', 'posix', 'symtable', 'nturl2path', 'array', 'dis', 'smtpd', 'sysconfig', 'grp', 'turtledemo', 'unicodedata', 'wave', 'timeit', 'warnings', 'this', 'subprocess', 'annotationlib', 'socket', 'zipapp', 'compileall', 'pprint', 'sre_constants', 'pdb', 'itertools', 'locale', 'mimetypes', 'codecs', 'macpath', 'msilib', 'netrc', 'nis', 'logging', 'telnetlib', 'modulefinder', 'errno', 'trace', 'ossaudiodev', 'encodings', 'functools', 'spwd', 'zipfile', 'configparser', 'gc', 'runpy', '_dummy_thread', 'pyexpat', 'sys', 'webbrowser', 'sndhdr', 'msvcrt', 'http', 'hmac', 'concurrent', 'plistlib', 'abc', 'code', 'marshal', 'bz2', 'graphlib', 'socketserver', 'sunau', 'bdb', 'copyreg', 'typing', 'xdrlib', 'base64', 'dbm', 'zipimport', 'platform', 'xxsubtype', 'calendar', 'formatter', 'keyword', 'contextvars', 'xx', 'hashlib', 'json', 'tabnanny', 'fileinput', 'sched', 'chunk', 'pickle', 'fpectl', 'contextlib', 'quopri', 'turtle', 'statistics', 'stat', 'winsound', 'token', 'zoneinfo', 'parser', 'select', '_ast', 'xxlimited_35', 'reprlib', 'sqlite3', 'termios', 'pydoc_data', 'getopt', 'copy', 'xml', 'imghdr', 'asyncio', 'queue', 'email', 'tokenize', 'uuid', 'selectors', 'shelve', 'math', 'test', 'inspect', 'atexit', 'profile', 'readline', 'stringprep', 'filecmp', 'pydoc', 'html', 'nntplib', 'poplib', 'threading', 'cProfile', 'ntpath', 'binhex', 'doctest', 'pickletools', 'types', 'audioop', 'ftplib', 'wsgiref', 'posixpath', 'csv', 'dummy_threading', 'heapq', 'time', 'weakref', 'operator', 'ctypes', '_thread', 'lib2to3', 'enum', 'tomllib', 'ssl', 'cgi', 'cgitb', 'mailcap', 'shlex', 'idlelib', 'smtplib', 'pipes', 'importlib', 'textwrap', 'fnmatch', 'struct', 'bisect', 'winreg', 'argparse', 'asynchat', 'shutil', 'tty', 'io', 'xmlrpc', 'nt', 'tkinter', 're', 'lzma', 'asyncore', 'faulthandler', 'resource', 'tracemalloc', 'distutils', 'symbol', 'fcntl', 'getpass', 'py_compile', 'linecache', 'fractions', 'imaplib', 'uu', 'unittest', 'xxlimited', 'multiprocessing', 'sre_compile', 'difflib', 'gettext'}), 'extra_standard_library': frozenset(), 'known_other': {}, 'multi_line_output': <WrapModes.GRID: 0>, 'forced_separate': (), 'indent': '    ', 'comment_prefix': '  #', 'length_sort': False, 'length_sort_straight': False, 'length_sort_sections': frozenset(), 'add_imports': frozenset(), 'remove_imports': frozenset(), 'append_only': False, 'reverse_relative': False, 'force_single_line': False, 'single_line_exclusions': (), 'default_section': 'THIRDPARTY', 'import_headings': {}, 'import_footers': {}, 'balanced_wrapping': False, 'use_parentheses': False, 'order_by_type': True, 'atomic': False, 'lines_before_imports': -1, 'lines_after_imports': -1, 'lines_between_sections': 1, 'lines_between_types': 0, 'combine_as_imports': False, 'combine_star': False, 'include_trailing_comma': False, 'from_first': False, 'verbose': False, 'quiet': False, 'force_adds': False, 'force_alphabetical_sort_within_sections': False, 'force_alphabetical_sort': False, 'force_grid_wrap': 0, 'force_sort_within_sections': False, 'lexicographical': False, 'group_by_package': False, 'ignore_whitespace': False, 'no_lines_before': frozenset(), 'no_inline_sort': False, 'ignore_comments': False, 'case_sensitive': False, 'sources': (), 'virtual_env': '', 'conda_env': '', 'ensure_newline_before_comments': False, 'directory': '', 'profile': '', 'honor_noqa': False, 'src_paths': (), 'remove_redundant_aliases': False, 'float_to_top': False, 'filter_files': False, 'formatter': '', 'formatting_function': None, 'color_output': False, 'treat_comments_as_code': frozenset(), 'treat_all_comments_as_code': False, 'supported_extensions': frozenset({'pyi', 'pxd', 'py', 'pyx'}), 'blocked_extensions': frozenset({'pex'}), 'constants': frozenset(), 'classes': frozenset(), 'variables': frozenset(), 'dedup_headings': False, 'only_sections': False, 'only_modified': False, 'combine_straight_imports': False, 'auto_identify_namespace_packages': True, 'namespace_packages': frozenset(), 'follow_links': True, 'indented_import_headings': True, 'honor_case_in_force_sorted_sections': False, 'sort_relative_in_force_sorted_sections': False, 'overwrite_in_place': False, 'reverse_sort': False, 'star_first': False, 'git_ls_files': {}, 'format_error': '{error}: {message}', 'format_success': '{success}: {message}', 'sort_order': 'natural', 'sort_reexports': False, 'split_on_trailing_comma': False, 'source': 'defaults'},), virtual_env='', conda_env='', ensure_newline_before_comments=False, directory='/workspace', profile='', honor_noqa=False, src_paths=(PosixPath('/workspace/src'), PosixPath('/workspace')), remove_redundant_aliases=False, float_to_top=False, filter_files=False, formatter='', formatting_function=None, color_output=False, treat_comments_as_code=frozenset(), treat_all_comments_as_code=False, supported_extensions=frozenset({'pyi', 'pxd', 'py', 'pyx'}), blocked_extensions=frozenset({'pex'}), constants=frozenset(), classes=frozenset(), variables=frozenset(), dedup_headings=False, only_sections=False, only_modified=False, combine_straight_imports=False, auto_identify_namespace_packages=True, namespace_packages=frozenset(), follow_links=True, indented_import_headings=True, honor_case_in_force_sorted_sections=False, sort_relative_in_force_sorted_sections=False, overwrite_in_place=False, reverse_sort=False, star_first=False, git_ls_files={}, format_error='{error}: {message}', format_success='{success}: {message}', sort_order='natural', sort_reexports=False, split_on_trailing_comma=False)"
    var_4 = module_0.find(var_3, var_1, var_3, var_2)
    with pytest.raises(AttributeError):
        var_5 = list(var_4)