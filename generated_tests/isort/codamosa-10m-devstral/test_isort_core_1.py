# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import urllib.parse as module_1
import urllib.request as module_2
import isort.wrap_modes as module_3
import email._header_value_parser as module_4
import encodings.idna as module_5
import _io as module_6
import isort.settings as module_7

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = False
    module_0.process(var_0, var_0, raise_on_skip=var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_1.unwrap(var_0)
    assert var_1 == 'None'
    assert module_1.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_1.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_1.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_1.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_1.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_1.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_1.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_1.MAX_CACHE_SIZE == 20
    module_0.process(var_1, var_1, raise_on_skip=var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_2.noheaders()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'email.message.Message'
    assert len(var_1) == 0
    assert module_2.MAXFTPCACHE == 10
    assert module_2.ftpcache == {}
    var_2 = module_0.process(var_1, var_0)
    assert var_2 is False
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
    assert module_0.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
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
    var_2.visit_Assert(var_1)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_4.quote_string(var_0)
    assert var_1 == '"None"'
    assert module_4.hexdigits == '0123456789abcdefABCDEF'
    assert module_4.WSP == {' ', '\t'}
    assert module_4.CFWS_LEADER == {' ', '\t', '('}
    assert module_4.SPECIALS == {'>', ':', '"', '(', ';', '@', '[', ']', '<', ',', '\\', ')', '.'}
    assert module_4.ATOM_ENDS == {'>', ':', '"', '(', ';', '@', '[', ']', ' ', '<', ',', '\t', '\\', ')', '.'}
    assert module_4.DOT_ATOM_ENDS == {'>', ':', '"', '(', ';', '@', '[', ']', ' ', '<', ',', '\t', '\\', ')'}
    assert module_4.PHRASE_ENDS == {'>', ':', ']', ';', '[', '@', '<', ',', '\\', ')'}
    assert module_4.TSPECIALS == {'=', '?', ':', '>', ']', '(', ';', '[', '"', '@', '<', ',', '\\', ')', '/'}
    assert module_4.TOKEN_ENDS == {'=', '?', ':', ']', '>', '(', ';', '[', '"', '@', ' ', '<', ',', '\t', '\\', ')', '/'}
    assert module_4.ASPECIALS == {'=', '?', ':', ']', '>', '(', ';', '[', '"', '@', '*', '<', ',', "'", '\\', ')', '/', '%'}
    assert module_4.ATTRIBUTE_ENDS == {':', '=', '>', ']', '@', ' ', '<', '\t', '\\', '%', '?', '(', ';', '[', '*', ',', ')', '"', "'", '/'}
    assert module_4.EXTENDED_ATTRIBUTE_ENDS == {':', '=', '>', ']', '@', ' ', '<', '\t', '\\', '?', '(', ';', '[', '*', ',', ')', '"', "'", '/'}
    assert module_4.NLSET == {'\r', '\n'}
    assert module_4.SPECIALSNL == {'\r', ':', '>', '"', '(', ';', '@', '[', ']', '\n', '<', ',', '\\', ')', '.'}
    assert f'{type(module_4.rfc2047_matcher).__module__}.{type(module_4.rfc2047_matcher).__qualname__}' == 're.Pattern'
    assert f'{type(module_4.DOT).__module__}.{type(module_4.DOT).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_4.DOT) == 1
    assert f'{type(module_4.ListSeparator).__module__}.{type(module_4.ListSeparator).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_4.ListSeparator) == 1
    assert f'{type(module_4.RouteComponentMarker).__module__}.{type(module_4.RouteComponentMarker).__qualname__}' == 'email._header_value_parser.ValueTerminal'
    assert len(module_4.RouteComponentMarker) == 1
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_5.getregentry()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'codecs.CodecInfo'
    assert len(var_0) == 4
    assert f'{type(module_5.unicodedata).__module__}.{type(module_5.unicodedata).__qualname__}' == 'unicodedata.UCD'
    assert f'{type(module_5.dots).__module__}.{type(module_5.dots).__qualname__}' == 're.Pattern'
    assert module_5.ace_prefix == b'xn--'
    assert module_5.sace_prefix == 'xn--'
    var_1 = var_0.__str__()
    var_2 = var_1.__repr__()
    var_3 = 'H1~'
    module_0.process(var_2, var_0, var_3)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_1.urlparse(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'urllib.parse.ParseResultBytes'
    assert len(var_1) == 6
    assert module_1.uses_relative == ['', 'ftp', 'http', 'gopher', 'nntp', 'imap', 'wais', 'file', 'https', 'shttp', 'mms', 'prospero', 'rtsp', 'rtspu', 'sftp', 'svn', 'svn+ssh', 'ws', 'wss']
    assert module_1.uses_netloc == ['', 'ftp', 'http', 'gopher', 'nntp', 'telnet', 'imap', 'wais', 'file', 'mms', 'https', 'shttp', 'snews', 'prospero', 'rtsp', 'rtspu', 'rsync', 'svn', 'svn+ssh', 'sftp', 'nfs', 'git', 'git+ssh', 'ws', 'wss']
    assert module_1.uses_params == ['', 'ftp', 'hdl', 'prospero', 'http', 'imap', 'https', 'shttp', 'rtsp', 'rtspu', 'sip', 'sips', 'mms', 'sftp', 'tel']
    assert module_1.non_hierarchical == ['gopher', 'hdl', 'mailto', 'news', 'telnet', 'wais', 'imap', 'snews', 'sip', 'sips']
    assert module_1.uses_query == ['', 'http', 'wais', 'imap', 'https', 'shttp', 'mms', 'gopher', 'rtsp', 'rtspu', 'sip', 'sips']
    assert module_1.uses_fragment == ['', 'ftp', 'hdl', 'http', 'gopher', 'news', 'nntp', 'wais', 'https', 'shttp', 'snews', 'file', 'prospero']
    assert module_1.scheme_chars == 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.'
    assert module_1.MAX_CACHE_SIZE == 20
    module_0.process(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_6.StringIO()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == '_io.StringIO'
    assert module_6.DEFAULT_BUFFER_SIZE == 8192
    assert f'{type(module_6.StringIO.closed).__module__}.{type(module_6.StringIO.closed).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.newlines).__module__}.{type(module_6.StringIO.newlines).__qualname__}' == 'builtins.getset_descriptor'
    assert f'{type(module_6.StringIO.line_buffering).__module__}.{type(module_6.StringIO.line_buffering).__qualname__}' == 'builtins.getset_descriptor'
    var_1 = module_6.StringIO()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == '_io.StringIO'
    var_2 = 'import z'
    var_3 = [var_2]
    var_4 = module_7.Config()
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
    assert var_4.multi_line_output == module_3.WrapModes.GRID
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
    assert module_7.TYPE_CHECKING is False
    assert module_7.profiles == {'black': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88}, 'django': {'combine_as_imports': True, 'include_trailing_comma': True, 'multi_line_output': 5, 'line_length': 79}, 'pycharm': {'multi_line_output': 3, 'force_grid_wrap': 2, 'lines_after_imports': 2}, 'google': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True, 'line_length': 1000, 'single_line_exclusions': ('collections.abc', 'six.moves', 'typing', 'typing_extensions'), 'order_by_type': False, 'group_by_package': True}, 'open_stack': {'force_single_line': True, 'force_sort_within_sections': True, 'lexicographical': True}, 'plone': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_alphabetical_sort': True, 'force_single_line': True, 'lines_after_imports': 2}, 'attrs': {'atomic': True, 'force_grid_wrap': 0, 'include_trailing_comma': True, 'lines_after_imports': 2, 'lines_between_types': 1, 'multi_line_output': 3, 'use_parentheses': True}, 'hug': {'multi_line_output': 3, 'include_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'line_length': 100}, 'wemake': {'multi_line_output': 3, 'include_trailing_comma': True, 'use_parentheses': True, 'line_length': 80}, 'appnexus': {'multi_line_output': 3, 'include_trailing_comma': True, 'split_on_trailing_comma': True, 'force_grid_wrap': 0, 'use_parentheses': True, 'ensure_newline_before_comments': True, 'line_length': 88, 'force_sort_within_sections': True, 'order_by_type': False, 'case_sensitive': False, 'reverse_relative': True, 'sort_relative_in_force_sorted_sections': True, 'sections': ['FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'APPLICATION', 'LOCALFOLDER'], 'no_lines_before': 'LOCALFOLDER'}}
    assert module_7.SECTION_DEFAULTS == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_7.FIRSTPARTY == 'FIRSTPARTY'
    assert module_7.FUTURE == 'FUTURE'
    assert module_7.LOCALFOLDER == 'LOCALFOLDER'
    assert module_7.STDLIB == 'STDLIB'
    assert module_7.THIRDPARTY == 'THIRDPARTY'
    assert f'{type(module_7.CYTHON_EXTENSIONS).__module__}.{type(module_7.CYTHON_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.CYTHON_EXTENSIONS) == 2
    assert f'{type(module_7.SUPPORTED_EXTENSIONS).__module__}.{type(module_7.SUPPORTED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.SUPPORTED_EXTENSIONS) == 4
    assert f'{type(module_7.BLOCKED_EXTENSIONS).__module__}.{type(module_7.BLOCKED_EXTENSIONS).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.BLOCKED_EXTENSIONS) == 1
    assert module_7.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_7.MAX_CONFIG_SEARCH_DEPTH == 25
    assert module_7.STOP_CONFIG_SEARCH_ON_DIRS == ('.git', '.hg')
    assert module_7.VALID_PY_TARGETS == ('all', '2', '27', '3', '310', '311', '312', '313', '314', '36', '37', '38', '39')
    assert module_7.CONFIG_SOURCES == ('.isort.cfg', 'pyproject.toml', 'setup.cfg', 'tox.ini', '.editorconfig')
    assert f'{type(module_7.DEFAULT_SKIP).__module__}.{type(module_7.DEFAULT_SKIP).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_SKIP) == 19
    assert module_7.CONFIG_SECTIONS == {'.isort.cfg': ('settings', 'isort'), 'pyproject.toml': ('tool.isort',), 'setup.cfg': ('isort', 'tool:isort'), 'tox.ini': ('isort', 'tool:isort'), '.editorconfig': ('*', '*.py', '**.py', '*.{py}')}
    assert module_7.FALLBACK_CONFIG_SECTIONS == ('isort', 'tool:isort', 'tool.isort')
    assert module_7.IMPORT_HEADING_PREFIX == 'import_heading_'
    assert module_7.IMPORT_FOOTER_PREFIX == 'import_footer_'
    assert module_7.KNOWN_PREFIX == 'known_'
    assert module_7.KNOWN_SECTION_MAPPING == {'STDLIB': 'STANDARD_LIBRARY', 'FUTURE': 'FUTURE_LIBRARY', 'FIRSTPARTY': 'FIRST_PARTY', 'THIRDPARTY': 'THIRD_PARTY', 'LOCALFOLDER': 'LOCAL_FOLDER'}
    assert module_7.RUNTIME_SOURCE == 'runtime'
    assert module_7.DEPRECATED_SETTINGS == ('not_skip', 'keep_direct_and_as_imports')
    assert f'{type(module_7.DEFAULT_CONFIG).__module__}.{type(module_7.DEFAULT_CONFIG).__qualname__}' == 'isort.settings.Config'
    assert module_7.DEFAULT_CONFIG.py_version == 'py3'
    assert f'{type(module_7.DEFAULT_CONFIG.force_to_top).__module__}.{type(module_7.DEFAULT_CONFIG.force_to_top).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.force_to_top) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.skip).__module__}.{type(module_7.DEFAULT_CONFIG.skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.skip) == 19
    assert f'{type(module_7.DEFAULT_CONFIG.extend_skip).__module__}.{type(module_7.DEFAULT_CONFIG.extend_skip).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.extend_skip) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.skip_glob).__module__}.{type(module_7.DEFAULT_CONFIG.skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.skip_glob) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.extend_skip_glob).__module__}.{type(module_7.DEFAULT_CONFIG.extend_skip_glob).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.extend_skip_glob) == 0
    assert module_7.DEFAULT_CONFIG.skip_gitignore is False
    assert module_7.DEFAULT_CONFIG.line_length == 79
    assert module_7.DEFAULT_CONFIG.wrap_length == 0
    assert module_7.DEFAULT_CONFIG.line_ending == ''
    assert module_7.DEFAULT_CONFIG.sections == ('FUTURE', 'STDLIB', 'THIRDPARTY', 'FIRSTPARTY', 'LOCALFOLDER')
    assert module_7.DEFAULT_CONFIG.no_sections is False
    assert f'{type(module_7.DEFAULT_CONFIG.known_future_library).__module__}.{type(module_7.DEFAULT_CONFIG.known_future_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.known_future_library) == 1
    assert f'{type(module_7.DEFAULT_CONFIG.known_third_party).__module__}.{type(module_7.DEFAULT_CONFIG.known_third_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.known_third_party) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.known_first_party).__module__}.{type(module_7.DEFAULT_CONFIG.known_first_party).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.known_first_party) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.known_local_folder).__module__}.{type(module_7.DEFAULT_CONFIG.known_local_folder).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.known_local_folder) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.known_standard_library).__module__}.{type(module_7.DEFAULT_CONFIG.known_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.known_standard_library) == 234
    assert f'{type(module_7.DEFAULT_CONFIG.extra_standard_library).__module__}.{type(module_7.DEFAULT_CONFIG.extra_standard_library).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.extra_standard_library) == 0
    assert module_7.DEFAULT_CONFIG.known_other == {}
    assert module_7.DEFAULT_CONFIG.multi_line_output == module_3.WrapModes.GRID
    assert module_7.DEFAULT_CONFIG.forced_separate == ()
    assert module_7.DEFAULT_CONFIG.indent == '    '
    assert module_7.DEFAULT_CONFIG.comment_prefix == '  #'
    assert module_7.DEFAULT_CONFIG.length_sort is False
    assert module_7.DEFAULT_CONFIG.length_sort_straight is False
    assert f'{type(module_7.DEFAULT_CONFIG.length_sort_sections).__module__}.{type(module_7.DEFAULT_CONFIG.length_sort_sections).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.length_sort_sections) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.add_imports).__module__}.{type(module_7.DEFAULT_CONFIG.add_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.add_imports) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.remove_imports).__module__}.{type(module_7.DEFAULT_CONFIG.remove_imports).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.remove_imports) == 0
    assert module_7.DEFAULT_CONFIG.append_only is False
    assert module_7.DEFAULT_CONFIG.reverse_relative is False
    assert module_7.DEFAULT_CONFIG.force_single_line is False
    assert module_7.DEFAULT_CONFIG.single_line_exclusions == ()
    assert module_7.DEFAULT_CONFIG.default_section == 'THIRDPARTY'
    assert module_7.DEFAULT_CONFIG.import_headings == {}
    assert module_7.DEFAULT_CONFIG.import_footers == {}
    assert module_7.DEFAULT_CONFIG.balanced_wrapping is False
    assert module_7.DEFAULT_CONFIG.use_parentheses is False
    assert module_7.DEFAULT_CONFIG.order_by_type is True
    assert module_7.DEFAULT_CONFIG.atomic is False
    assert module_7.DEFAULT_CONFIG.lines_before_imports == -1
    assert module_7.DEFAULT_CONFIG.lines_after_imports == -1
    assert module_7.DEFAULT_CONFIG.lines_between_sections == 1
    assert module_7.DEFAULT_CONFIG.lines_between_types == 0
    assert module_7.DEFAULT_CONFIG.combine_as_imports is False
    assert module_7.DEFAULT_CONFIG.combine_star is False
    assert module_7.DEFAULT_CONFIG.include_trailing_comma is False
    assert module_7.DEFAULT_CONFIG.from_first is False
    assert module_7.DEFAULT_CONFIG.verbose is False
    assert module_7.DEFAULT_CONFIG.quiet is False
    assert module_7.DEFAULT_CONFIG.force_adds is False
    assert module_7.DEFAULT_CONFIG.force_alphabetical_sort_within_sections is False
    assert module_7.DEFAULT_CONFIG.force_alphabetical_sort is False
    assert module_7.DEFAULT_CONFIG.force_grid_wrap == 0
    assert module_7.DEFAULT_CONFIG.force_sort_within_sections is False
    assert module_7.DEFAULT_CONFIG.lexicographical is False
    assert module_7.DEFAULT_CONFIG.group_by_package is False
    assert module_7.DEFAULT_CONFIG.ignore_whitespace is False
    assert f'{type(module_7.DEFAULT_CONFIG.no_lines_before).__module__}.{type(module_7.DEFAULT_CONFIG.no_lines_before).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.no_lines_before) == 0
    assert module_7.DEFAULT_CONFIG.no_inline_sort is False
    assert module_7.DEFAULT_CONFIG.ignore_comments is False
    assert module_7.DEFAULT_CONFIG.case_sensitive is False
    assert f'{type(module_7.DEFAULT_CONFIG.sources).__module__}.{type(module_7.DEFAULT_CONFIG.sources).__qualname__}' == 'builtins.tuple'
    assert len(module_7.DEFAULT_CONFIG.sources) == 1
    assert module_7.DEFAULT_CONFIG.virtual_env == ''
    assert module_7.DEFAULT_CONFIG.conda_env == ''
    assert module_7.DEFAULT_CONFIG.ensure_newline_before_comments is False
    assert module_7.DEFAULT_CONFIG.directory == '/workspace'
    assert module_7.DEFAULT_CONFIG.profile == ''
    assert module_7.DEFAULT_CONFIG.honor_noqa is False
    assert f'{type(module_7.DEFAULT_CONFIG.src_paths).__module__}.{type(module_7.DEFAULT_CONFIG.src_paths).__qualname__}' == 'builtins.tuple'
    assert len(module_7.DEFAULT_CONFIG.src_paths) == 2
    assert module_7.DEFAULT_CONFIG.remove_redundant_aliases is False
    assert module_7.DEFAULT_CONFIG.float_to_top is False
    assert module_7.DEFAULT_CONFIG.filter_files is False
    assert module_7.DEFAULT_CONFIG.formatter == ''
    assert module_7.DEFAULT_CONFIG.formatting_function is None
    assert module_7.DEFAULT_CONFIG.color_output is False
    assert f'{type(module_7.DEFAULT_CONFIG.treat_comments_as_code).__module__}.{type(module_7.DEFAULT_CONFIG.treat_comments_as_code).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.treat_comments_as_code) == 0
    assert module_7.DEFAULT_CONFIG.treat_all_comments_as_code is False
    assert f'{type(module_7.DEFAULT_CONFIG.supported_extensions).__module__}.{type(module_7.DEFAULT_CONFIG.supported_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.supported_extensions) == 4
    assert f'{type(module_7.DEFAULT_CONFIG.blocked_extensions).__module__}.{type(module_7.DEFAULT_CONFIG.blocked_extensions).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.blocked_extensions) == 1
    assert f'{type(module_7.DEFAULT_CONFIG.constants).__module__}.{type(module_7.DEFAULT_CONFIG.constants).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.constants) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.classes).__module__}.{type(module_7.DEFAULT_CONFIG.classes).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.classes) == 0
    assert f'{type(module_7.DEFAULT_CONFIG.variables).__module__}.{type(module_7.DEFAULT_CONFIG.variables).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.variables) == 0
    assert module_7.DEFAULT_CONFIG.dedup_headings is False
    assert module_7.DEFAULT_CONFIG.only_sections is False
    assert module_7.DEFAULT_CONFIG.only_modified is False
    assert module_7.DEFAULT_CONFIG.combine_straight_imports is False
    assert module_7.DEFAULT_CONFIG.auto_identify_namespace_packages is True
    assert f'{type(module_7.DEFAULT_CONFIG.namespace_packages).__module__}.{type(module_7.DEFAULT_CONFIG.namespace_packages).__qualname__}' == 'builtins.frozenset'
    assert len(module_7.DEFAULT_CONFIG.namespace_packages) == 0
    assert module_7.DEFAULT_CONFIG.follow_links is True
    assert module_7.DEFAULT_CONFIG.indented_import_headings is True
    assert module_7.DEFAULT_CONFIG.honor_case_in_force_sorted_sections is False
    assert module_7.DEFAULT_CONFIG.sort_relative_in_force_sorted_sections is False
    assert module_7.DEFAULT_CONFIG.overwrite_in_place is False
    assert module_7.DEFAULT_CONFIG.reverse_sort is False
    assert module_7.DEFAULT_CONFIG.star_first is False
    assert module_7.DEFAULT_CONFIG.git_ls_files == {}
    assert module_7.DEFAULT_CONFIG.format_error == '{error}: {message}'
    assert module_7.DEFAULT_CONFIG.format_success == '{success}: {message}'
    assert module_7.DEFAULT_CONFIG.sort_order == 'natural'
    assert module_7.DEFAULT_CONFIG.sort_reexports is False
    assert module_7.DEFAULT_CONFIG.split_on_trailing_comma is False
    assert f'{type(module_7.Config.known_patterns).__module__}.{type(module_7.Config.known_patterns).__qualname__}' == 'builtins.property'
    assert f'{type(module_7.Config.section_comments).__module__}.{type(module_7.Config.section_comments).__qualname__}' == 'builtins.property'
    assert f'{type(module_7.Config.section_comments_end).__module__}.{type(module_7.Config.section_comments_end).__qualname__}' == 'builtins.property'
    assert f'{type(module_7.Config.skips).__module__}.{type(module_7.Config.skips).__qualname__}' == 'builtins.property'
    assert f'{type(module_7.Config.skip_globs).__module__}.{type(module_7.Config.skip_globs).__qualname__}' == 'builtins.property'
    assert f'{type(module_7.Config.sorting_function).__module__}.{type(module_7.Config.sorting_function).__qualname__}' == 'builtins.property'
    var_5 = module_6.StringIO()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == '_io.StringIO'
    var_6 = '# isort: off\nimport b\nimport a\n# isort: on\nimport c\n'
    var_7 = module_6.StringIO()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == '_io.StringIO'
    var_8 = module_6.StringIO()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == '_io.StringIO'
    var_9 = module_6.StringIO()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == '_io.StringIO'
    var_10 = module_6.StringIO()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == '_io.StringIO'
    var_11 = module_6.StringIO()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == '_io.StringIO'
    var_12 = None
    module_0.process(var_6, var_12)