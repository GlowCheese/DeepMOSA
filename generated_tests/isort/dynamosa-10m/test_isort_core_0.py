# Check out: https://github.com/GlowCheese/deepmosa
import email._header_value_parser as module_2
import genericpath as module_3
import importlib.metadata as module_1
import urllib.request as module_4

import isort.core as module_0
import isort.parse as module_6
import isort.wrap_modes as module_5
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.process(var_0, var_0, config=var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_1.packages_distributions()
    module_0.process(var_0, var_0, raise_on_skip=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_1.packages_distributions()
    var_2 = module_2.quote_string(var_0)
    assert var_2 == '"None"'
    assert module_2.hexdigits == '0123456789abcdefABCDEF'
    assert module_2.WSP == {'\t', ' '}
    assert module_2.CFWS_LEADER == {'(', '\t', ' '}
    assert module_2.SPECIALS == {'>', '(', ';', ')', '.', ']', '\\', ':', '[', '@', ',', '<', '"'}
    assert module_2.ATOM_ENDS == {'>', '(', ' ', ';', ')', '.', ']', '\\', ':', '[', '\t', '@', ',', '<', '"'}
    assert module_2.DOT_ATOM_ENDS == {'>', '(', ' ', ';', ')', ']', '\\', ':', '[', '\t', '@', ',', '<', '"'}
    assert module_2.PHRASE_ENDS == {')', ';', '<', ']', '\\', ':', '[', '@', ',', '>'}
    assert module_2.TSPECIALS == {'(', ')', ';', '<', ']', '\\', '/', ':', '[', '@', '?', ',', '>', '=', '"'}
    assert module_2.TOKEN_ENDS == {'>', '(', ' ', ';', ')', ']', '\\', '/', ':', '[', '@', '\t', '?', ',', '<', '=', '"'}
    assert module_2.ASPECIALS == {'>', '(', '=', ';', ')', ']', '\\', '*', '/', ':', '[', '@', "'", '?', ',', '<', '%', '"'}
    assert module_2.ATTRIBUTE_ENDS == {']', '*', '\t', '?', ',', '>', '%', ' ', '\\', ':', '<', ')', '/', '@', '=', '(', ';', '[', "'", '"'}
    assert module_2.EXTENDED_ATTRIBUTE_ENDS == {']', '*', '\t', '?', ',', '>', ' ', '\\', ':', '<', ')', '/', '@', '=', '(', ';', '[', "'", '"'}
    assert module_2.NLSET == {'\n', '\r'}
    assert module_2.SPECIALSNL == {'>', '(', '\r', ';', ')', '.', ']', '\\', ':', '[', '@', '\n', '<', '"', ','}
    assert f'{type(module_2.rfc2047_matcher).__module__}.{type(module_2.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_2.DOT).__module__}.{type(module_2.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.DOT) == 1
    assert f'{type(module_2.ListSeparator).__module__}.{type(module_2.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.ListSeparator) == 1
    assert f'{type(module_2.RouteComponentMarker).__module__}.{type(module_2.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_2.RouteComponentMarker) == 1
    module_0.process(var_2, var_2, raise_on_skip=var_2)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_3.commonprefix(var_0)
    assert var_1 == ''
    assert f'{type(module_3.ALLOW_MISSING).__module__}.{type(module_3.ALLOW_MISSING).__qualname__}' == 'genericpath.ALLOW_MISSING'
    var_2 = var_1.__repr__()
    assert var_2 == "''"
    module_0.process(var_2, var_1)

def test_case_5():
    var_0 = module_4.getproxies_environment()
    assert module_4.MAXFTPCACHE == 10
    assert module_4.ftpcache == {}
    var_1 = module_0.process(var_0, var_0, raise_on_skip=var_0)
    assert var_1 is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_5.WrapModes.GRID
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
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ' pm\rx$D6'
    module_0.process(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = ":wt/ O'9V"
    var_1 = module_6.normalize_line(var_0)
    assert module_6.TYPE_CHECKING is False
    assert f'{type(module_6.DEFAULT_CONFIG).__module__}.{type(module_6.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_6.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_6.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_6.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip).__module__}.{type(module_6.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_6.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_6.DEFAULT_CONFIG.skip_gitignore is False
    assert module_6.DEFAULT_CONFIG.line_length == 79
    assert module_6.DEFAULT_CONFIG.wrap_length == 0
    assert module_6.DEFAULT_CONFIG.line_ending == ''
    assert module_6.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_6.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_6.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_6.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_6.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_6.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_6.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_6.DEFAULT_CONFIG.known_other == {}
    assert module_6.DEFAULT_CONFIG.multi_line_output == module_5.WrapModes.GRID
    assert module_6.DEFAULT_CONFIG.forced_separate == ()
    assert module_6.DEFAULT_CONFIG.indent == '    '
    assert module_6.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_6.DEFAULT_CONFIG.length_sort is False
    assert module_6.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_6.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_6.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.add_imports).__module__}.{type(module_6.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_6.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.remove_imports) == 0
    assert module_6.DEFAULT_CONFIG.append_only is False
    assert module_6.DEFAULT_CONFIG.reverse_relative is False
    assert module_6.DEFAULT_CONFIG.force_single_line is False
    assert module_6.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_6.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_6.DEFAULT_CONFIG.import_headings == {}
    assert module_6.DEFAULT_CONFIG.import_footers == {}
    assert module_6.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_6.DEFAULT_CONFIG.use_parentheses is False
    assert module_6.DEFAULT_CONFIG.order_by_type is True
    assert module_6.DEFAULT_CONFIG.atomic is False
    assert module_6.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_6.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_6.DEFAULT_CONFIG.lines_between_types == 0
    assert module_6.DEFAULT_CONFIG.combine_as_imports is False
    assert module_6.DEFAULT_CONFIG.combine_star is False
    assert module_6.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_6.DEFAULT_CONFIG.from_first is False
    assert module_6.DEFAULT_CONFIG.verbose is False
    assert module_6.DEFAULT_CONFIG.quiet is False
    assert module_6.DEFAULT_CONFIG.force_adds is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_6.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_6.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_6.DEFAULT_CONFIG.lexicographical is False
    assert module_6.DEFAULT_CONFIG.group_by_package is False
    assert module_6.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_6.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_6.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_6.DEFAULT_CONFIG.no_inline_sort is False
    assert module_6.DEFAULT_CONFIG.ignore_comments is False
    assert module_6.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_6.DEFAULT_CONFIG.sources).__module__}.{type(module_6.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.sources) == 1
    assert module_6.DEFAULT_CONFIG.virtual_env == ''
    assert module_6.DEFAULT_CONFIG.conda_env == ''
    assert module_6.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_6.DEFAULT_CONFIG.directory == '/workspace'
    assert module_6.DEFAULT_CONFIG.profile == ''
    assert module_6.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_6.DEFAULT_CONFIG.src_paths).__module__}.{type(module_6.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_6.DEFAULT_CONFIG.src_paths) == 2
    assert module_6.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_6.DEFAULT_CONFIG.float_to_top is False
    assert module_6.DEFAULT_CONFIG.filter_files is False
    assert module_6.DEFAULT_CONFIG.formatter == ''
    assert module_6.DEFAULT_CONFIG.formatting_function is None
    assert module_6.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_6.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_6.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_6.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_6.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_6.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_6.DEFAULT_CONFIG.constants).__module__}.{type(module_6.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.classes).__module__}.{type(module_6.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_6.DEFAULT_CONFIG.variables).__module__}.{type(module_6.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.variables) == 0
    assert module_6.DEFAULT_CONFIG.dedup_headings is False
    assert module_6.DEFAULT_CONFIG.only_sections is False
    assert module_6.DEFAULT_CONFIG.only_modified is False
    assert module_6.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_6.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_6.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_6.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_6.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_6.DEFAULT_CONFIG.follow_links is True
    assert module_6.DEFAULT_CONFIG.indented_import_headings is True
    assert module_6.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_6.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_6.DEFAULT_CONFIG.reverse_sort is False
    assert module_6.DEFAULT_CONFIG.star_first is False
    assert module_6.DEFAULT_CONFIG.git_ls_files == {}
    assert module_6.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_6.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_6.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_6.DEFAULT_CONFIG.sort_reexports is False
    assert module_6.DEFAULT_CONFIG.split_on_trailing_comma is False
    module_0.process(var_1, var_1)