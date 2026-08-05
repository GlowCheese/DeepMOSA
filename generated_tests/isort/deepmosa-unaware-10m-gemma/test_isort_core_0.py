# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.core as module_0
import locale as module_1
import urllib.request as module_2
import isort.wrap_modes as module_3
import http.cookies as module_4

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.process(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.getpreferredencoding()
    assert var_0 == 'UTF-8'
    assert module_1.LC_CTYPE == 0
    assert module_1.LC_TIME == 2
    assert module_1.LC_COLLATE == 3
    assert module_1.LC_MONETARY == 4
    assert module_1.LC_MESSAGES == 5
    assert module_1.LC_NUMERIC == 1
    assert module_1.LC_ALL == 6
    assert module_1.CHAR_MAX == 127
    assert module_1.DAY_1 == 131079
    assert module_1.DAY_2 == 131080
    assert module_1.DAY_3 == 131081
    assert module_1.DAY_4 == 131082
    assert module_1.DAY_5 == 131083
    assert module_1.DAY_6 == 131084
    assert module_1.DAY_7 == 131085
    assert module_1.ABDAY_1 == 131072
    assert module_1.ABDAY_2 == 131073
    assert module_1.ABDAY_3 == 131074
    assert module_1.ABDAY_4 == 131075
    assert module_1.ABDAY_5 == 131076
    assert module_1.ABDAY_6 == 131077
    assert module_1.ABDAY_7 == 131078
    assert module_1.MON_1 == 131098
    assert module_1.MON_2 == 131099
    assert module_1.MON_3 == 131100
    assert module_1.MON_4 == 131101
    assert module_1.MON_5 == 131102
    assert module_1.MON_6 == 131103
    assert module_1.MON_7 == 131104
    assert module_1.MON_8 == 131105
    assert module_1.MON_9 == 131106
    assert module_1.MON_10 == 131107
    assert module_1.MON_11 == 131108
    assert module_1.MON_12 == 131109
    assert module_1.ABMON_1 == 131086
    assert module_1.ABMON_2 == 131087
    assert module_1.ABMON_3 == 131088
    assert module_1.ABMON_4 == 131089
    assert module_1.ABMON_5 == 131090
    assert module_1.ABMON_6 == 131091
    assert module_1.ABMON_7 == 131092
    assert module_1.ABMON_8 == 131093
    assert module_1.ABMON_9 == 131094
    assert module_1.ABMON_10 == 131095
    assert module_1.ABMON_11 == 131096
    assert module_1.ABMON_12 == 131097
    assert module_1.RADIXCHAR == 65536
    assert module_1.THOUSEP == 65537
    assert module_1.CRNCYSTR == 262159
    assert module_1.D_T_FMT == 131112
    assert module_1.D_FMT == 131113
    assert module_1.T_FMT == 131114
    assert module_1.AM_STR == 131110
    assert module_1.PM_STR == 131111
    assert module_1.CODESET == 14
    assert module_1.T_FMT_AMPM == 131115
    assert module_1.ERA == 131116
    assert module_1.ERA_D_FMT == 131118
    assert module_1.ERA_D_T_FMT == 131120
    assert module_1.ERA_T_FMT == 131121
    assert module_1.ALT_DIGITS == 131119
    assert module_1.YESEXPR == 327680
    assert module_1.NOEXPR == 327681
    assert module_1.locale_encoding_alias == {'437': 'C', 'c': 'C', 'en': 'ISO8859-1', 'jis': 'JIS7', 'jis7': 'JIS7', 'ajec': 'eucJP', 'koi8c': 'KOI8-C', 'microsoftcp1251': 'CP1251', 'microsoftcp1255': 'CP1255', 'microsoftcp1256': 'CP1256', '88591': 'ISO8859-1', '88592': 'ISO8859-2', '88595': 'ISO8859-5', '885915': 'ISO8859-15', 'ascii': 'ISO8859-1', 'latin_1': 'ISO8859-1', 'iso8859_1': 'ISO8859-1', 'iso8859_10': 'ISO8859-10', 'iso8859_11': 'ISO8859-11', 'iso8859_13': 'ISO8859-13', 'iso8859_14': 'ISO8859-14', 'iso8859_15': 'ISO8859-15', 'iso8859_16': 'ISO8859-16', 'iso8859_2': 'ISO8859-2', 'iso8859_3': 'ISO8859-3', 'iso8859_4': 'ISO8859-4', 'iso8859_5': 'ISO8859-5', 'iso8859_6': 'ISO8859-6', 'iso8859_7': 'ISO8859-7', 'iso8859_8': 'ISO8859-8', 'iso8859_9': 'ISO8859-9', 'iso2022_jp': 'JIS7', 'shift_jis': 'SJIS', 'tactis': 'TACTIS', 'euc_jp': 'eucJP', 'euc_kr': 'eucKR', 'utf_8': 'UTF-8', 'koi8_r': 'KOI8-R', 'koi8_t': 'KOI8-T', 'koi8_u': 'KOI8-U', 'kz1048': 'RK1048', 'cp1251': 'CP1251', 'cp1255': 'CP1255', 'cp1256': 'CP1256', 'eucjp': 'eucJP', 'euckr': 'eucKR', 'iso2022jp': 'JIS7', 'iso88591': 'ISO8859-1', 'iso885910': 'ISO8859-10', 'iso885911': 'ISO8859-11', 'iso885913': 'ISO8859-13', 'iso885914': 'ISO8859-14', 'iso885915': 'ISO8859-15', 'iso885916': 'ISO8859-16', 'iso88592': 'ISO8859-2', 'iso88593': 'ISO8859-3', 'iso88594': 'ISO8859-4', 'iso88595': 'ISO8859-5', 'iso88596': 'ISO8859-6', 'iso88597': 'ISO8859-7', 'iso88598': 'ISO8859-8', 'iso88599': 'ISO8859-9', 'koi8r': 'KOI8-R', 'koi8t': 'KOI8-T', 'koi8u': 'KOI8-U', 'latin1': 'ISO8859-1', 'shiftjis': 'SJIS', 'utf8': 'UTF-8'}
    assert module_1.k == 'utf8'
    assert module_1.v == 'UTF-8'
    assert module_1.locale_alias == {'a3': 'az_AZ.KOI8-C', 'a3_az': 'az_AZ.KOI8-C', 'a3_az.koic': 'az_AZ.KOI8-C', 'aa_dj': 'aa_DJ.ISO8859-1', 'aa_er': 'aa_ER.UTF-8', 'aa_et': 'aa_ET.UTF-8', 'af': 'af_ZA.ISO8859-1', 'af_za': 'af_ZA.ISO8859-1', 'agr_pe': 'agr_PE.UTF-8', 'ak_gh': 'ak_GH.UTF-8', 'am': 'am_ET.UTF-8', 'am_et': 'am_ET.UTF-8', 'american': 'en_US.ISO8859-1', 'an_es': 'an_ES.ISO8859-15', 'anp_in': 'anp_IN.UTF-8', 'ar': 'ar_AA.ISO8859-6', 'ar_aa': 'ar_AA.ISO8859-6', 'ar_ae': 'ar_AE.ISO8859-6', 'ar_bh': 'ar_BH.ISO8859-6', 'ar_dz': 'ar_DZ.ISO8859-6', 'ar_eg': 'ar_EG.ISO8859-6', 'ar_in': 'ar_IN.UTF-8', 'ar_iq': 'ar_IQ.ISO8859-6', 'ar_jo': 'ar_JO.ISO8859-6', 'ar_kw': 'ar_KW.ISO8859-6', 'ar_lb': 'ar_LB.ISO8859-6', 'ar_ly': 'ar_LY.ISO8859-6', 'ar_ma': 'ar_MA.ISO8859-6', 'ar_om': 'ar_OM.ISO8859-6', 'ar_qa': 'ar_QA.ISO8859-6', 'ar_sa': 'ar_SA.ISO8859-6', 'ar_sd': 'ar_SD.ISO8859-6', 'ar_ss': 'ar_SS.UTF-8', 'ar_sy': 'ar_SY.ISO8859-6', 'ar_tn': 'ar_TN.ISO8859-6', 'ar_ye': 'ar_YE.ISO8859-6', 'arabic': 'ar_AA.ISO8859-6', 'as': 'as_IN.UTF-8', 'as_in': 'as_IN.UTF-8', 'ast_es': 'ast_ES.ISO8859-15', 'ayc_pe': 'ayc_PE.UTF-8', 'az': 'az_AZ.ISO8859-9E', 'az_az': 'az_AZ.ISO8859-9E', 'az_az.iso88599e': 'az_AZ.ISO8859-9E', 'az_ir': 'az_IR.UTF-8', 'be': 'be_BY.CP1251', 'be@latin': 'be_BY.UTF-8@latin', 'be_bg.utf8': 'bg_BG.UTF-8', 'be_by': 'be_BY.CP1251', 'be_by@latin': 'be_BY.UTF-8@latin', 'bem_zm': 'bem_ZM.UTF-8', 'ber_dz': 'ber_DZ.UTF-8', 'ber_ma': 'ber_MA.UTF-8', 'bg': 'bg_BG.CP1251', 'bg_bg': 'bg_BG.CP1251', 'bhb_in.utf8': 'bhb_IN.UTF-8', 'bho_in': 'bho_IN.UTF-8', 'bho_np': 'bho_NP.UTF-8', 'bi_vu': 'bi_VU.UTF-8', 'bn_bd': 'bn_BD.UTF-8', 'bn_in': 'bn_IN.UTF-8', 'bo_cn': 'bo_CN.UTF-8', 'bo_in': 'bo_IN.UTF-8', 'bokmal': 'nb_NO.ISO8859-1', 'bokmål': 'nb_NO.ISO8859-1', 'br': 'br_FR.ISO8859-1', 'br_fr': 'br_FR.ISO8859-1', 'brx_in': 'brx_IN.UTF-8', 'bs': 'bs_BA.ISO8859-2', 'bs_ba': 'bs_BA.ISO8859-2', 'bulgarian': 'bg_BG.CP1251', 'byn_er': 'byn_ER.UTF-8', 'c': 'C', 'c-french': 'fr_CA.ISO8859-1', 'c.ascii': 'C', 'c.en': 'C', 'c.iso88591': 'en_US.ISO8859-1', 'c.utf8': 'en_US.UTF-8', 'c_c': 'C', 'c_c.c': 'C', 'ca': 'ca_ES.ISO8859-1', 'ca_ad': 'ca_AD.ISO8859-1', 'ca_es': 'ca_ES.ISO8859-1', 'ca_es@valencia': 'ca_ES.UTF-8@valencia', 'ca_fr': 'ca_FR.ISO8859-1', 'ca_it': 'ca_IT.ISO8859-1', 'catalan': 'ca_ES.ISO8859-1', 'ce_ru': 'ce_RU.UTF-8', 'cextend': 'en_US.ISO8859-1', 'chinese-s': 'zh_CN.eucCN', 'chinese-t': 'zh_TW.eucTW', 'chr_us': 'chr_US.UTF-8', 'ckb_iq': 'ckb_IQ.UTF-8', 'cmn_tw': 'cmn_TW.UTF-8', 'crh_ua': 'crh_UA.UTF-8', 'croatian': 'hr_HR.ISO8859-2', 'cs': 'cs_CZ.ISO8859-2', 'cs_cs': 'cs_CZ.ISO8859-2', 'cs_cz': 'cs_CZ.ISO8859-2', 'csb_pl': 'csb_PL.UTF-8', 'cv_ru': 'cv_RU.UTF-8', 'cy': 'cy_GB.ISO8859-1', 'cy_gb': 'cy_GB.ISO8859-1', 'cz': 'cs_CZ.ISO8859-2', 'cz_cz': 'cs_CZ.ISO8859-2', 'czech': 'cs_CZ.ISO8859-2', 'da': 'da_DK.ISO8859-1', 'da_dk': 'da_DK.ISO8859-1', 'danish': 'da_DK.ISO8859-1', 'dansk': 'da_DK.ISO8859-1', 'de': 'de_DE.ISO8859-1', 'de_at': 'de_AT.ISO8859-1', 'de_be': 'de_BE.ISO8859-1', 'de_ch': 'de_CH.ISO8859-1', 'de_de': 'de_DE.ISO8859-1', 'de_it': 'de_IT.ISO8859-1', 'de_li.utf8': 'de_LI.UTF-8', 'de_lu': 'de_LU.ISO8859-1', 'deutsch': 'de_DE.ISO8859-1', 'doi_in': 'doi_IN.UTF-8', 'dutch': 'nl_NL.ISO8859-1', 'dutch.iso88591': 'nl_BE.ISO8859-1', 'dv_mv': 'dv_MV.UTF-8', 'dz_bt': 'dz_BT.UTF-8', 'ee': 'ee_EE.ISO8859-4', 'ee_ee': 'ee_EE.ISO8859-4', 'eesti': 'et_EE.ISO8859-1', 'el': 'el_GR.ISO8859-7', 'el_cy': 'el_CY.ISO8859-7', 'el_gr': 'el_GR.ISO8859-7', 'el_gr@euro': 'el_GR.ISO8859-15', 'en': 'en_US.ISO8859-1', 'en_ag': 'en_AG.UTF-8', 'en_au': 'en_AU.ISO8859-1', 'en_be': 'en_BE.ISO8859-1', 'en_bw': 'en_BW.ISO8859-1', 'en_ca': 'en_CA.ISO8859-1', 'en_dk': 'en_DK.ISO8859-1', 'en_dl.utf8': 'en_DL.UTF-8', 'en_gb': 'en_GB.ISO8859-1', 'en_hk': 'en_HK.ISO8859-1', 'en_ie': 'en_IE.ISO8859-1', 'en_il': 'en_IL.UTF-8', 'en_in': 'en_IN.ISO8859-1', 'en_ng': 'en_NG.UTF-8', 'en_nz': 'en_NZ.ISO8859-1', 'en_ph': 'en_PH.ISO8859-1', 'en_sc.utf8': 'en_SC.UTF-8', 'en_sg': 'en_SG.ISO8859-1', 'en_uk': 'en_GB.ISO8859-1', 'en_us': 'en_US.ISO8859-1', 'en_us@euro@euro': 'en_US.ISO8859-15', 'en_za': 'en_ZA.ISO8859-1', 'en_zm': 'en_ZM.UTF-8', 'en_zw': 'en_ZW.ISO8859-1', 'en_zw.utf8': 'en_ZS.UTF-8', 'eng_gb': 'en_GB.ISO8859-1', 'english': 'en_EN.ISO8859-1', 'english.iso88591': 'en_US.ISO8859-1', 'english_uk': 'en_GB.ISO8859-1', 'english_united-states': 'en_US.ISO8859-1', 'english_united-states.437': 'C', 'english_us': 'en_US.ISO8859-1', 'eo': 'eo_XX.ISO8859-3', 'eo.utf8': 'eo.UTF-8', 'eo_eo': 'eo_EO.ISO8859-3', 'eo_us.utf8': 'eo_US.UTF-8', 'eo_xx': 'eo_XX.ISO8859-3', 'es': 'es_ES.ISO8859-1', 'es_ar': 'es_AR.ISO8859-1', 'es_bo': 'es_BO.ISO8859-1', 'es_cl': 'es_CL.ISO8859-1', 'es_co': 'es_CO.ISO8859-1', 'es_cr': 'es_CR.ISO8859-1', 'es_cu': 'es_CU.UTF-8', 'es_do': 'es_DO.ISO8859-1', 'es_ec': 'es_EC.ISO8859-1', 'es_es': 'es_ES.ISO8859-1', 'es_gt': 'es_GT.ISO8859-1', 'es_hn': 'es_HN.ISO8859-1', 'es_mx': 'es_MX.ISO8859-1', 'es_ni': 'es_NI.ISO8859-1', 'es_pa': 'es_PA.ISO8859-1', 'es_pe': 'es_PE.ISO8859-1', 'es_pr': 'es_PR.ISO8859-1', 'es_py': 'es_PY.ISO8859-1', 'es_sv': 'es_SV.ISO8859-1', 'es_us': 'es_US.ISO8859-1', 'es_uy': 'es_UY.ISO8859-1', 'es_ve': 'es_VE.ISO8859-1', 'estonian': 'et_EE.ISO8859-1', 'et': 'et_EE.ISO8859-15', 'et_ee': 'et_EE.ISO8859-15', 'eu': 'eu_ES.ISO8859-1', 'eu_es': 'eu_ES.ISO8859-1', 'eu_fr': 'eu_FR.ISO8859-1', 'fa': 'fa_IR.UTF-8', 'fa_ir': 'fa_IR.UTF-8', 'fa_ir.isiri3342': 'fa_IR.ISIRI-3342', 'ff_sn': 'ff_SN.UTF-8', 'fi': 'fi_FI.ISO8859-15', 'fi_fi': 'fi_FI.ISO8859-15', 'fil_ph': 'fil_PH.UTF-8', 'finnish': 'fi_FI.ISO8859-1', 'fo': 'fo_FO.ISO8859-1', 'fo_fo': 'fo_FO.ISO8859-1', 'fr': 'fr_FR.ISO8859-1', 'fr_be': 'fr_BE.ISO8859-1', 'fr_ca': 'fr_CA.ISO8859-1', 'fr_ch': 'fr_CH.ISO8859-1', 'fr_fr': 'fr_FR.ISO8859-1', 'fr_lu': 'fr_LU.ISO8859-1', 'français': 'fr_FR.ISO8859-1', 'fre_fr': 'fr_FR.ISO8859-1', 'french': 'fr_FR.ISO8859-1', 'french.iso88591': 'fr_CH.ISO8859-1', 'french_france': 'fr_FR.ISO8859-1', 'fur_it': 'fur_IT.UTF-8', 'fy_de': 'fy_DE.UTF-8', 'fy_nl': 'fy_NL.UTF-8', 'ga': 'ga_IE.ISO8859-1', 'ga_ie': 'ga_IE.ISO8859-1', 'galego': 'gl_ES.ISO8859-1', 'galician': 'gl_ES.ISO8859-1', 'gd': 'gd_GB.ISO8859-1', 'gd_gb': 'gd_GB.ISO8859-1', 'ger_de': 'de_DE.ISO8859-1', 'german': 'de_DE.ISO8859-1', 'german.iso88591': 'de_CH.ISO8859-1', 'german_germany': 'de_DE.ISO8859-1', 'gez_er': 'gez_ER.UTF-8', 'gez_et': 'gez_ET.UTF-8', 'gl': 'gl_ES.ISO8859-1', 'gl_es': 'gl_ES.ISO8859-1', 'greek': 'el_GR.ISO8859-7', 'gu_in': 'gu_IN.UTF-8', 'gv': 'gv_GB.ISO8859-1', 'gv_gb': 'gv_GB.ISO8859-1', 'ha_ng': 'ha_NG.UTF-8', 'hak_tw': 'hak_TW.UTF-8', 'he': 'he_IL.ISO8859-8', 'he_il': 'he_IL.ISO8859-8', 'hebrew': 'he_IL.ISO8859-8', 'hi': 'hi_IN.ISCII-DEV', 'hi_in': 'hi_IN.ISCII-DEV', 'hi_in.isciidev': 'hi_IN.ISCII-DEV', 'hif_fj': 'hif_FJ.UTF-8', 'hne': 'hne_IN.UTF-8', 'hne_in': 'hne_IN.UTF-8', 'hr': 'hr_HR.ISO8859-2', 'hr_hr': 'hr_HR.ISO8859-2', 'hrvatski': 'hr_HR.ISO8859-2', 'hsb_de': 'hsb_DE.ISO8859-2', 'ht_ht': 'ht_HT.UTF-8', 'hu': 'hu_HU.ISO8859-2', 'hu_hu': 'hu_HU.ISO8859-2', 'hungarian': 'hu_HU.ISO8859-2', 'hy_am': 'hy_AM.UTF-8', 'hy_am.armscii8': 'hy_AM.ARMSCII_8', 'ia': 'ia.UTF-8', 'ia_fr': 'ia_FR.UTF-8', 'icelandic': 'is_IS.ISO8859-1', 'id': 'id_ID.ISO8859-1', 'id_id': 'id_ID.ISO8859-1', 'ig_ng': 'ig_NG.UTF-8', 'ik_ca': 'ik_CA.UTF-8', 'in': 'id_ID.ISO8859-1', 'in_id': 'id_ID.ISO8859-1', 'is': 'is_IS.ISO8859-1', 'is_is': 'is_IS.ISO8859-1', 'iso-8859-1': 'en_US.ISO8859-1', 'iso-8859-15': 'en_US.ISO8859-15', 'iso8859-1': 'en_US.ISO8859-1', 'iso8859-15': 'en_US.ISO8859-15', 'iso_8859_1': 'en_US.ISO8859-1', 'iso_8859_15': 'en_US.ISO8859-15', 'it': 'it_IT.ISO8859-1', 'it_ch': 'it_CH.ISO8859-1', 'it_it': 'it_IT.ISO8859-1', 'italian': 'it_IT.ISO8859-1', 'iu': 'iu_CA.NUNACOM-8', 'iu_ca': 'iu_CA.NUNACOM-8', 'iu_ca.nunacom8': 'iu_CA.NUNACOM-8', 'iw': 'he_IL.ISO8859-8', 'iw_il': 'he_IL.ISO8859-8', 'iw_il.utf8': 'iw_IL.UTF-8', 'ja': 'ja_JP.eucJP', 'ja_jp': 'ja_JP.eucJP', 'ja_jp.euc': 'ja_JP.eucJP', 'ja_jp.mscode': 'ja_JP.SJIS', 'ja_jp.pck': 'ja_JP.SJIS', 'japan': 'ja_JP.eucJP', 'japanese': 'ja_JP.eucJP', 'japanese-euc': 'ja_JP.eucJP', 'japanese.euc': 'ja_JP.eucJP', 'jp_jp': 'ja_JP.eucJP', 'ka': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge.georgianacademy': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge.georgianps': 'ka_GE.GEORGIAN-PS', 'ka_ge.georgianrs': 'ka_GE.GEORGIAN-ACADEMY', 'kab_dz': 'kab_DZ.UTF-8', 'kk_kz': 'kk_KZ.ptcp154', 'kl': 'kl_GL.ISO8859-1', 'kl_gl': 'kl_GL.ISO8859-1', 'km_kh': 'km_KH.UTF-8', 'kn': 'kn_IN.UTF-8', 'kn_in': 'kn_IN.UTF-8', 'ko': 'ko_KR.eucKR', 'ko_kr': 'ko_KR.eucKR', 'ko_kr.euc': 'ko_KR.eucKR', 'kok_in': 'kok_IN.UTF-8', 'korean': 'ko_KR.eucKR', 'korean.euc': 'ko_KR.eucKR', 'ks': 'ks_IN.UTF-8', 'ks_in': 'ks_IN.UTF-8', 'ks_in@devanagari.utf8': 'ks_IN.UTF-8@devanagari', 'ku_tr': 'ku_TR.ISO8859-9', 'kw': 'kw_GB.ISO8859-1', 'kw_gb': 'kw_GB.ISO8859-1', 'ky': 'ky_KG.UTF-8', 'ky_kg': 'ky_KG.UTF-8', 'lb_lu': 'lb_LU.UTF-8', 'lg_ug': 'lg_UG.ISO8859-10', 'li_be': 'li_BE.UTF-8', 'li_nl': 'li_NL.UTF-8', 'lij_it': 'lij_IT.UTF-8', 'lithuanian': 'lt_LT.ISO8859-13', 'ln_cd': 'ln_CD.UTF-8', 'lo': 'lo_LA.MULELAO-1', 'lo_la': 'lo_LA.MULELAO-1', 'lo_la.cp1133': 'lo_LA.IBM-CP1133', 'lo_la.ibmcp1133': 'lo_LA.IBM-CP1133', 'lo_la.mulelao1': 'lo_LA.MULELAO-1', 'lt': 'lt_LT.ISO8859-13', 'lt_lt': 'lt_LT.ISO8859-13', 'lv': 'lv_LV.ISO8859-13', 'lv_lv': 'lv_LV.ISO8859-13', 'lzh_tw': 'lzh_TW.UTF-8', 'mag_in': 'mag_IN.UTF-8', 'mai': 'mai_IN.UTF-8', 'mai_in': 'mai_IN.UTF-8', 'mai_np': 'mai_NP.UTF-8', 'mfe_mu': 'mfe_MU.UTF-8', 'mg_mg': 'mg_MG.ISO8859-15', 'mhr_ru': 'mhr_RU.UTF-8', 'mi': 'mi_NZ.ISO8859-1', 'mi_nz': 'mi_NZ.ISO8859-1', 'miq_ni': 'miq_NI.UTF-8', 'mjw_in': 'mjw_IN.UTF-8', 'mk': 'mk_MK.ISO8859-5', 'mk_mk': 'mk_MK.ISO8859-5', 'ml': 'ml_IN.UTF-8', 'ml_in': 'ml_IN.UTF-8', 'mn_mn': 'mn_MN.UTF-8', 'mni_in': 'mni_IN.UTF-8', 'mr': 'mr_IN.UTF-8', 'mr_in': 'mr_IN.UTF-8', 'ms': 'ms_MY.ISO8859-1', 'ms_my': 'ms_MY.ISO8859-1', 'mt': 'mt_MT.ISO8859-3', 'mt_mt': 'mt_MT.ISO8859-3', 'my_mm': 'my_MM.UTF-8', 'nan_tw': 'nan_TW.UTF-8', 'nb': 'nb_NO.ISO8859-1', 'nb_no': 'nb_NO.ISO8859-1', 'nds_de': 'nds_DE.UTF-8', 'nds_nl': 'nds_NL.UTF-8', 'ne_np': 'ne_NP.UTF-8', 'nhn_mx': 'nhn_MX.UTF-8', 'niu_nu': 'niu_NU.UTF-8', 'niu_nz': 'niu_NZ.UTF-8', 'nl': 'nl_NL.ISO8859-1', 'nl_aw': 'nl_AW.UTF-8', 'nl_be': 'nl_BE.ISO8859-1', 'nl_nl': 'nl_NL.ISO8859-1', 'nn': 'nn_NO.ISO8859-1', 'nn_no': 'nn_NO.ISO8859-1', 'no': 'no_NO.ISO8859-1', 'no@nynorsk': 'ny_NO.ISO8859-1', 'no_no': 'no_NO.ISO8859-1', 'no_no.iso88591@bokmal': 'no_NO.ISO8859-1', 'no_no.iso88591@nynorsk': 'no_NO.ISO8859-1', 'norwegian': 'no_NO.ISO8859-1', 'nr': 'nr_ZA.ISO8859-1', 'nr_za': 'nr_ZA.ISO8859-1', 'nso': 'nso_ZA.ISO8859-15', 'nso_za': 'nso_ZA.ISO8859-15', 'ny': 'ny_NO.ISO8859-1', 'ny_no': 'ny_NO.ISO8859-1', 'nynorsk': 'nn_NO.ISO8859-1', 'oc': 'oc_FR.ISO8859-1', 'oc_fr': 'oc_FR.ISO8859-1', 'om_et': 'om_ET.UTF-8', 'om_ke': 'om_KE.ISO8859-1', 'or': 'or_IN.UTF-8', 'or_in': 'or_IN.UTF-8', 'os_ru': 'os_RU.UTF-8', 'pa': 'pa_IN.UTF-8', 'pa_in': 'pa_IN.UTF-8', 'pa_pk': 'pa_PK.UTF-8', 'pap_an': 'pap_AN.UTF-8', 'pap_aw': 'pap_AW.UTF-8', 'pap_cw': 'pap_CW.UTF-8', 'pd': 'pd_US.ISO8859-1', 'pd_de': 'pd_DE.ISO8859-1', 'pd_us': 'pd_US.ISO8859-1', 'ph': 'ph_PH.ISO8859-1', 'ph_ph': 'ph_PH.ISO8859-1', 'pl': 'pl_PL.ISO8859-2', 'pl_pl': 'pl_PL.ISO8859-2', 'polish': 'pl_PL.ISO8859-2', 'portuguese': 'pt_PT.ISO8859-1', 'portuguese_brazil': 'pt_BR.ISO8859-1', 'posix': 'C', 'posix-utf2': 'C', 'pp': 'pp_AN.ISO8859-1', 'pp_an': 'pp_AN.ISO8859-1', 'ps_af': 'ps_AF.UTF-8', 'pt': 'pt_PT.ISO8859-1', 'pt_br': 'pt_BR.ISO8859-1', 'pt_pt': 'pt_PT.ISO8859-1', 'quz_pe': 'quz_PE.UTF-8', 'raj_in': 'raj_IN.UTF-8', 'ro': 'ro_RO.ISO8859-2', 'ro_ro': 'ro_RO.ISO8859-2', 'romanian': 'ro_RO.ISO8859-2', 'ru': 'ru_RU.UTF-8', 'ru_ru': 'ru_RU.UTF-8', 'ru_ua': 'ru_UA.KOI8-U', 'rumanian': 'ro_RO.ISO8859-2', 'russian': 'ru_RU.KOI8-R', 'rw': 'rw_RW.ISO8859-1', 'rw_rw': 'rw_RW.ISO8859-1', 'sa_in': 'sa_IN.UTF-8', 'sat_in': 'sat_IN.UTF-8', 'sc_it': 'sc_IT.UTF-8', 'sd': 'sd_IN.UTF-8', 'sd_in': 'sd_IN.UTF-8', 'sd_in@devanagari.utf8': 'sd_IN.UTF-8@devanagari', 'sd_pk': 'sd_PK.UTF-8', 'se_no': 'se_NO.UTF-8', 'serbocroatian': 'sr_RS.UTF-8@latin', 'sgs_lt': 'sgs_LT.UTF-8', 'sh': 'sr_RS.UTF-8@latin', 'sh_ba.iso88592@bosnia': 'sr_CS.ISO8859-2', 'sh_hr': 'sh_HR.ISO8859-2', 'sh_hr.iso88592': 'hr_HR.ISO8859-2', 'sh_sp': 'sr_CS.ISO8859-2', 'sh_yu': 'sr_RS.UTF-8@latin', 'shn_mm': 'shn_MM.UTF-8', 'shs_ca': 'shs_CA.UTF-8', 'si': 'si_LK.UTF-8', 'si_lk': 'si_LK.UTF-8', 'sid_et': 'sid_ET.UTF-8', 'sinhala': 'si_LK.UTF-8', 'sk': 'sk_SK.ISO8859-2', 'sk_sk': 'sk_SK.ISO8859-2', 'sl': 'sl_SI.ISO8859-2', 'sl_cs': 'sl_CS.ISO8859-2', 'sl_si': 'sl_SI.ISO8859-2', 'slovak': 'sk_SK.ISO8859-2', 'slovene': 'sl_SI.ISO8859-2', 'slovenian': 'sl_SI.ISO8859-2', 'sm_ws': 'sm_WS.UTF-8', 'so_dj': 'so_DJ.ISO8859-1', 'so_et': 'so_ET.UTF-8', 'so_ke': 'so_KE.ISO8859-1', 'so_so': 'so_SO.ISO8859-1', 'sp': 'sr_CS.ISO8859-5', 'sp_yu': 'sr_CS.ISO8859-5', 'spanish': 'es_ES.ISO8859-1', 'spanish_spain': 'es_ES.ISO8859-1', 'sq': 'sq_AL.ISO8859-2', 'sq_al': 'sq_AL.ISO8859-2', 'sq_mk': 'sq_MK.UTF-8', 'sr': 'sr_RS.UTF-8', 'sr@cyrillic': 'sr_RS.UTF-8', 'sr@latn': 'sr_CS.UTF-8@latin', 'sr_cs': 'sr_CS.UTF-8', 'sr_cs.iso88592@latn': 'sr_CS.ISO8859-2', 'sr_cs@latn': 'sr_CS.UTF-8@latin', 'sr_me': 'sr_ME.UTF-8', 'sr_rs': 'sr_RS.UTF-8', 'sr_rs@latn': 'sr_RS.UTF-8@latin', 'sr_sp': 'sr_CS.ISO8859-2', 'sr_yu': 'sr_RS.UTF-8@latin', 'sr_yu.cp1251@cyrillic': 'sr_CS.CP1251', 'sr_yu.iso88592': 'sr_CS.ISO8859-2', 'sr_yu.iso88595': 'sr_CS.ISO8859-5', 'sr_yu.iso88595@cyrillic': 'sr_CS.ISO8859-5', 'sr_yu.microsoftcp1251@cyrillic': 'sr_CS.CP1251', 'sr_yu.utf8': 'sr_RS.UTF-8', 'sr_yu.utf8@cyrillic': 'sr_RS.UTF-8', 'sr_yu@cyrillic': 'sr_RS.UTF-8', 'ss': 'ss_ZA.ISO8859-1', 'ss_za': 'ss_ZA.ISO8859-1', 'st': 'st_ZA.ISO8859-1', 'st_za': 'st_ZA.ISO8859-1', 'sv': 'sv_SE.ISO8859-1', 'sv_fi': 'sv_FI.ISO8859-1', 'sv_se': 'sv_SE.ISO8859-1', 'sw_ke': 'sw_KE.UTF-8', 'sw_tz': 'sw_TZ.UTF-8', 'swedish': 'sv_SE.ISO8859-1', 'szl_pl': 'szl_PL.UTF-8', 'ta': 'ta_IN.TSCII-0', 'ta_in': 'ta_IN.TSCII-0', 'ta_in.tscii': 'ta_IN.TSCII-0', 'ta_in.tscii0': 'ta_IN.TSCII-0', 'ta_lk': 'ta_LK.UTF-8', 'tcy_in.utf8': 'tcy_IN.UTF-8', 'te': 'te_IN.UTF-8', 'te_in': 'te_IN.UTF-8', 'tg': 'tg_TJ.KOI8-C', 'tg_tj': 'tg_TJ.KOI8-C', 'th': 'th_TH.ISO8859-11', 'th_th': 'th_TH.ISO8859-11', 'th_th.tactis': 'th_TH.TIS620', 'th_th.tis620': 'th_TH.TIS620', 'thai': 'th_TH.ISO8859-11', 'the_np': 'the_NP.UTF-8', 'ti_er': 'ti_ER.UTF-8', 'ti_et': 'ti_ET.UTF-8', 'tig_er': 'tig_ER.UTF-8', 'tk_tm': 'tk_TM.UTF-8', 'tl': 'tl_PH.ISO8859-1', 'tl_ph': 'tl_PH.ISO8859-1', 'tn': 'tn_ZA.ISO8859-15', 'tn_za': 'tn_ZA.ISO8859-15', 'to_to': 'to_TO.UTF-8', 'tpi_pg': 'tpi_PG.UTF-8', 'tr': 'tr_TR.ISO8859-9', 'tr_cy': 'tr_CY.ISO8859-9', 'tr_tr': 'tr_TR.ISO8859-9', 'ts': 'ts_ZA.ISO8859-1', 'ts_za': 'ts_ZA.ISO8859-1', 'tt': 'tt_RU.TATAR-CYR', 'tt_ru': 'tt_RU.TATAR-CYR', 'tt_ru.tatarcyr': 'tt_RU.TATAR-CYR', 'tt_ru@iqtelif': 'tt_RU.UTF-8@iqtelif', 'turkish': 'tr_TR.ISO8859-9', 'ug_cn': 'ug_CN.UTF-8', 'uk': 'uk_UA.KOI8-U', 'uk_ua': 'uk_UA.KOI8-U', 'univ': 'en_US.utf', 'universal': 'en_US.utf', 'universal.utf8@ucs4': 'en_US.UTF-8', 'unm_us': 'unm_US.UTF-8', 'ur': 'ur_PK.CP1256', 'ur_in': 'ur_IN.UTF-8', 'ur_pk': 'ur_PK.CP1256', 'uz': 'uz_UZ.UTF-8', 'uz_uz': 'uz_UZ.UTF-8', 'uz_uz@cyrillic': 'uz_UZ.UTF-8', 've': 've_ZA.UTF-8', 've_za': 've_ZA.UTF-8', 'vi': 'vi_VN.TCVN', 'vi_vn': 'vi_VN.TCVN', 'vi_vn.tcvn': 'vi_VN.TCVN', 'vi_vn.tcvn5712': 'vi_VN.TCVN', 'vi_vn.viscii': 'vi_VN.VISCII', 'vi_vn.viscii111': 'vi_VN.VISCII', 'wa': 'wa_BE.ISO8859-1', 'wa_be': 'wa_BE.ISO8859-1', 'wae_ch': 'wae_CH.UTF-8', 'wal_et': 'wal_ET.UTF-8', 'wo_sn': 'wo_SN.UTF-8', 'xh': 'xh_ZA.ISO8859-1', 'xh_za': 'xh_ZA.ISO8859-1', 'yi': 'yi_US.CP1255', 'yi_us': 'yi_US.CP1255', 'yo_ng': 'yo_NG.UTF-8', 'yue_hk': 'yue_HK.UTF-8', 'yuw_pg': 'yuw_PG.UTF-8', 'zh': 'zh_CN.eucCN', 'zh_cn': 'zh_CN.gb2312', 'zh_cn.big5': 'zh_TW.big5', 'zh_cn.euc': 'zh_CN.eucCN', 'zh_hk': 'zh_HK.big5hkscs', 'zh_hk.big5hk': 'zh_HK.big5hkscs', 'zh_sg': 'zh_SG.GB2312', 'zh_sg.gbk': 'zh_SG.GBK', 'zh_tw': 'zh_TW.big5', 'zh_tw.euc': 'zh_TW.eucTW', 'zh_tw.euctw': 'zh_TW.eucTW', 'zu': 'zu_ZA.ISO8859-1', 'zu_za': 'zu_ZA.ISO8859-1'}
    assert module_1.windows_locale == {1078: 'af_ZA', 1052: 'sq_AL', 1156: 'gsw_FR', 1118: 'am_ET', 1025: 'ar_SA', 2049: 'ar_IQ', 3073: 'ar_EG', 4097: 'ar_LY', 5121: 'ar_DZ', 6145: 'ar_MA', 7169: 'ar_TN', 8193: 'ar_OM', 9217: 'ar_YE', 10241: 'ar_SY', 11265: 'ar_JO', 12289: 'ar_LB', 13313: 'ar_KW', 14337: 'ar_AE', 15361: 'ar_BH', 16385: 'ar_QA', 1067: 'hy_AM', 1101: 'as_IN', 1068: 'az_AZ', 2092: 'az_AZ', 1133: 'ba_RU', 1069: 'eu_ES', 1059: 'be_BY', 1093: 'bn_IN', 8218: 'bs_BA', 5146: 'bs_BA', 1150: 'br_FR', 1026: 'bg_BG', 1027: 'ca_ES', 4: 'zh_CHS', 1028: 'zh_TW', 2052: 'zh_CN', 3076: 'zh_HK', 4100: 'zh_SG', 5124: 'zh_MO', 31748: 'zh_CHT', 1155: 'co_FR', 1050: 'hr_HR', 4122: 'hr_BA', 1029: 'cs_CZ', 1030: 'da_DK', 1164: 'gbz_AF', 1125: 'div_MV', 1043: 'nl_NL', 2067: 'nl_BE', 1033: 'en_US', 2057: 'en_GB', 3081: 'en_AU', 4105: 'en_CA', 5129: 'en_NZ', 6153: 'en_IE', 7177: 'en_ZA', 8201: 'en_JA', 9225: 'en_CB', 10249: 'en_BZ', 11273: 'en_TT', 12297: 'en_ZW', 13321: 'en_PH', 16393: 'en_IN', 17417: 'en_MY', 18441: 'en_IN', 1061: 'et_EE', 1080: 'fo_FO', 1124: 'fil_PH', 1035: 'fi_FI', 1036: 'fr_FR', 2060: 'fr_BE', 3084: 'fr_CA', 4108: 'fr_CH', 5132: 'fr_LU', 6156: 'fr_MC', 1122: 'fy_NL', 1110: 'gl_ES', 1079: 'ka_GE', 1031: 'de_DE', 2055: 'de_CH', 3079: 'de_AT', 4103: 'de_LU', 5127: 'de_LI', 1032: 'el_GR', 1135: 'kl_GL', 1095: 'gu_IN', 1128: 'ha_NG', 1037: 'he_IL', 1081: 'hi_IN', 1038: 'hu_HU', 1039: 'is_IS', 1057: 'id_ID', 1117: 'iu_CA', 2141: 'iu_CA', 2108: 'ga_IE', 1040: 'it_IT', 2064: 'it_CH', 1041: 'ja_JP', 1099: 'kn_IN', 1087: 'kk_KZ', 1107: 'kh_KH', 1158: 'qut_GT', 1159: 'rw_RW', 1111: 'kok_IN', 1042: 'ko_KR', 1088: 'ky_KG', 1108: 'lo_LA', 1062: 'lv_LV', 1063: 'lt_LT', 2094: 'dsb_DE', 1134: 'lb_LU', 1071: 'mk_MK', 1086: 'ms_MY', 2110: 'ms_BN', 1100: 'ml_IN', 1082: 'mt_MT', 1153: 'mi_NZ', 1146: 'arn_CL', 1102: 'mr_IN', 1148: 'moh_CA', 1104: 'mn_MN', 2128: 'mn_CN', 1121: 'ne_NP', 1044: 'nb_NO', 2068: 'nn_NO', 1154: 'oc_FR', 1096: 'or_IN', 1123: 'ps_AF', 1065: 'fa_IR', 1045: 'pl_PL', 1046: 'pt_BR', 2070: 'pt_PT', 1094: 'pa_IN', 1131: 'quz_BO', 2155: 'quz_EC', 3179: 'quz_PE', 1048: 'ro_RO', 1047: 'rm_CH', 1049: 'ru_RU', 9275: 'smn_FI', 4155: 'smj_NO', 5179: 'smj_SE', 1083: 'se_NO', 2107: 'se_SE', 3131: 'se_FI', 8251: 'sms_FI', 6203: 'sma_NO', 7227: 'sma_SE', 1103: 'sa_IN', 3098: 'sr_SP', 7194: 'sr_BA', 2074: 'sr_SP', 6170: 'sr_BA', 1115: 'si_LK', 1132: 'ns_ZA', 1074: 'tn_ZA', 1051: 'sk_SK', 1060: 'sl_SI', 1034: 'es_ES', 2058: 'es_MX', 3082: 'es_ES', 4106: 'es_GT', 5130: 'es_CR', 6154: 'es_PA', 7178: 'es_DO', 8202: 'es_VE', 9226: 'es_CO', 10250: 'es_PE', 11274: 'es_AR', 12298: 'es_EC', 13322: 'es_CL', 14346: 'es_UR', 15370: 'es_PY', 16394: 'es_BO', 17418: 'es_SV', 18442: 'es_HN', 19466: 'es_NI', 20490: 'es_PR', 21514: 'es_US', 1089: 'sw_KE', 1053: 'sv_SE', 2077: 'sv_FI', 1114: 'syr_SY', 1064: 'tg_TJ', 2143: 'tmz_DZ', 1097: 'ta_IN', 1092: 'tt_RU', 1098: 'te_IN', 1054: 'th_TH', 2129: 'bo_BT', 1105: 'bo_CN', 1055: 'tr_TR', 1090: 'tk_TM', 1152: 'ug_CN', 1058: 'uk_UA', 1070: 'wen_DE', 1056: 'ur_PK', 2080: 'ur_IN', 1091: 'uz_UZ', 2115: 'uz_UZ', 1066: 'vi_VN', 1106: 'cy_GB', 1160: 'wo_SN', 1076: 'xh_ZA', 1157: 'sah_RU', 1144: 'ii_CN', 1130: 'yo_NG', 1077: 'zu_ZA'}
    module_0.process(var_0, var_0)

def test_case_2():
    var_0 = module_2.noheaders()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'email.message.Message'
    assert len(var_0) == 0
    assert module_2.MAXFTPCACHE == 10
    assert module_2.ftpcache == {}
    var_1 = module_0.process(var_0, var_0)
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
    assert module_0.FILE_SKIP_COMMENTS == ('isort:skip_file', 'isort: skip_file')
    assert module_0.CIMPORT_IDENTIFIERS == ('cimport ', 'cimport*', 'from.cimport')
    assert module_0.IMPORT_START_IDENTIFIERS == ('from ', 'from.import', 'import ', 'import*', 'cimport ', 'cimport*', 'from.cimport')
    assert module_0.DOCSTRING_INDICATORS == ('"""', "'''")
    assert module_0.COMMENT_INDICATORS == ('"""', "'''", "'", '"', '#')
    assert module_0.CODE_SORT_COMMENTS == ('# isort: list', '# isort: dict', '# isort: set', '# isort: unique-list', '# isort: tuple', '# isort: unique-tuple', '# isort: assignments')
    assert module_0.LITERAL_TYPE_MAPPING == {'(': 'tuple', '[': 'list', '{': 'set'}

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = module_1.getpreferredencoding()
    assert var_0 == 'UTF-8'
    assert module_1.LC_CTYPE == 0
    assert module_1.LC_TIME == 2
    assert module_1.LC_COLLATE == 3
    assert module_1.LC_MONETARY == 4
    assert module_1.LC_MESSAGES == 5
    assert module_1.LC_NUMERIC == 1
    assert module_1.LC_ALL == 6
    assert module_1.CHAR_MAX == 127
    assert module_1.DAY_1 == 131079
    assert module_1.DAY_2 == 131080
    assert module_1.DAY_3 == 131081
    assert module_1.DAY_4 == 131082
    assert module_1.DAY_5 == 131083
    assert module_1.DAY_6 == 131084
    assert module_1.DAY_7 == 131085
    assert module_1.ABDAY_1 == 131072
    assert module_1.ABDAY_2 == 131073
    assert module_1.ABDAY_3 == 131074
    assert module_1.ABDAY_4 == 131075
    assert module_1.ABDAY_5 == 131076
    assert module_1.ABDAY_6 == 131077
    assert module_1.ABDAY_7 == 131078
    assert module_1.MON_1 == 131098
    assert module_1.MON_2 == 131099
    assert module_1.MON_3 == 131100
    assert module_1.MON_4 == 131101
    assert module_1.MON_5 == 131102
    assert module_1.MON_6 == 131103
    assert module_1.MON_7 == 131104
    assert module_1.MON_8 == 131105
    assert module_1.MON_9 == 131106
    assert module_1.MON_10 == 131107
    assert module_1.MON_11 == 131108
    assert module_1.MON_12 == 131109
    assert module_1.ABMON_1 == 131086
    assert module_1.ABMON_2 == 131087
    assert module_1.ABMON_3 == 131088
    assert module_1.ABMON_4 == 131089
    assert module_1.ABMON_5 == 131090
    assert module_1.ABMON_6 == 131091
    assert module_1.ABMON_7 == 131092
    assert module_1.ABMON_8 == 131093
    assert module_1.ABMON_9 == 131094
    assert module_1.ABMON_10 == 131095
    assert module_1.ABMON_11 == 131096
    assert module_1.ABMON_12 == 131097
    assert module_1.RADIXCHAR == 65536
    assert module_1.THOUSEP == 65537
    assert module_1.CRNCYSTR == 262159
    assert module_1.D_T_FMT == 131112
    assert module_1.D_FMT == 131113
    assert module_1.T_FMT == 131114
    assert module_1.AM_STR == 131110
    assert module_1.PM_STR == 131111
    assert module_1.CODESET == 14
    assert module_1.T_FMT_AMPM == 131115
    assert module_1.ERA == 131116
    assert module_1.ERA_D_FMT == 131118
    assert module_1.ERA_D_T_FMT == 131120
    assert module_1.ERA_T_FMT == 131121
    assert module_1.ALT_DIGITS == 131119
    assert module_1.YESEXPR == 327680
    assert module_1.NOEXPR == 327681
    assert module_1.locale_encoding_alias == {'437': 'C', 'c': 'C', 'en': 'ISO8859-1', 'jis': 'JIS7', 'jis7': 'JIS7', 'ajec': 'eucJP', 'koi8c': 'KOI8-C', 'microsoftcp1251': 'CP1251', 'microsoftcp1255': 'CP1255', 'microsoftcp1256': 'CP1256', '88591': 'ISO8859-1', '88592': 'ISO8859-2', '88595': 'ISO8859-5', '885915': 'ISO8859-15', 'ascii': 'ISO8859-1', 'latin_1': 'ISO8859-1', 'iso8859_1': 'ISO8859-1', 'iso8859_10': 'ISO8859-10', 'iso8859_11': 'ISO8859-11', 'iso8859_13': 'ISO8859-13', 'iso8859_14': 'ISO8859-14', 'iso8859_15': 'ISO8859-15', 'iso8859_16': 'ISO8859-16', 'iso8859_2': 'ISO8859-2', 'iso8859_3': 'ISO8859-3', 'iso8859_4': 'ISO8859-4', 'iso8859_5': 'ISO8859-5', 'iso8859_6': 'ISO8859-6', 'iso8859_7': 'ISO8859-7', 'iso8859_8': 'ISO8859-8', 'iso8859_9': 'ISO8859-9', 'iso2022_jp': 'JIS7', 'shift_jis': 'SJIS', 'tactis': 'TACTIS', 'euc_jp': 'eucJP', 'euc_kr': 'eucKR', 'utf_8': 'UTF-8', 'koi8_r': 'KOI8-R', 'koi8_t': 'KOI8-T', 'koi8_u': 'KOI8-U', 'kz1048': 'RK1048', 'cp1251': 'CP1251', 'cp1255': 'CP1255', 'cp1256': 'CP1256', 'eucjp': 'eucJP', 'euckr': 'eucKR', 'iso2022jp': 'JIS7', 'iso88591': 'ISO8859-1', 'iso885910': 'ISO8859-10', 'iso885911': 'ISO8859-11', 'iso885913': 'ISO8859-13', 'iso885914': 'ISO8859-14', 'iso885915': 'ISO8859-15', 'iso885916': 'ISO8859-16', 'iso88592': 'ISO8859-2', 'iso88593': 'ISO8859-3', 'iso88594': 'ISO8859-4', 'iso88595': 'ISO8859-5', 'iso88596': 'ISO8859-6', 'iso88597': 'ISO8859-7', 'iso88598': 'ISO8859-8', 'iso88599': 'ISO8859-9', 'koi8r': 'KOI8-R', 'koi8t': 'KOI8-T', 'koi8u': 'KOI8-U', 'latin1': 'ISO8859-1', 'shiftjis': 'SJIS', 'utf8': 'UTF-8'}
    assert module_1.k == 'utf8'
    assert module_1.v == 'UTF-8'
    assert module_1.locale_alias == {'a3': 'az_AZ.KOI8-C', 'a3_az': 'az_AZ.KOI8-C', 'a3_az.koic': 'az_AZ.KOI8-C', 'aa_dj': 'aa_DJ.ISO8859-1', 'aa_er': 'aa_ER.UTF-8', 'aa_et': 'aa_ET.UTF-8', 'af': 'af_ZA.ISO8859-1', 'af_za': 'af_ZA.ISO8859-1', 'agr_pe': 'agr_PE.UTF-8', 'ak_gh': 'ak_GH.UTF-8', 'am': 'am_ET.UTF-8', 'am_et': 'am_ET.UTF-8', 'american': 'en_US.ISO8859-1', 'an_es': 'an_ES.ISO8859-15', 'anp_in': 'anp_IN.UTF-8', 'ar': 'ar_AA.ISO8859-6', 'ar_aa': 'ar_AA.ISO8859-6', 'ar_ae': 'ar_AE.ISO8859-6', 'ar_bh': 'ar_BH.ISO8859-6', 'ar_dz': 'ar_DZ.ISO8859-6', 'ar_eg': 'ar_EG.ISO8859-6', 'ar_in': 'ar_IN.UTF-8', 'ar_iq': 'ar_IQ.ISO8859-6', 'ar_jo': 'ar_JO.ISO8859-6', 'ar_kw': 'ar_KW.ISO8859-6', 'ar_lb': 'ar_LB.ISO8859-6', 'ar_ly': 'ar_LY.ISO8859-6', 'ar_ma': 'ar_MA.ISO8859-6', 'ar_om': 'ar_OM.ISO8859-6', 'ar_qa': 'ar_QA.ISO8859-6', 'ar_sa': 'ar_SA.ISO8859-6', 'ar_sd': 'ar_SD.ISO8859-6', 'ar_ss': 'ar_SS.UTF-8', 'ar_sy': 'ar_SY.ISO8859-6', 'ar_tn': 'ar_TN.ISO8859-6', 'ar_ye': 'ar_YE.ISO8859-6', 'arabic': 'ar_AA.ISO8859-6', 'as': 'as_IN.UTF-8', 'as_in': 'as_IN.UTF-8', 'ast_es': 'ast_ES.ISO8859-15', 'ayc_pe': 'ayc_PE.UTF-8', 'az': 'az_AZ.ISO8859-9E', 'az_az': 'az_AZ.ISO8859-9E', 'az_az.iso88599e': 'az_AZ.ISO8859-9E', 'az_ir': 'az_IR.UTF-8', 'be': 'be_BY.CP1251', 'be@latin': 'be_BY.UTF-8@latin', 'be_bg.utf8': 'bg_BG.UTF-8', 'be_by': 'be_BY.CP1251', 'be_by@latin': 'be_BY.UTF-8@latin', 'bem_zm': 'bem_ZM.UTF-8', 'ber_dz': 'ber_DZ.UTF-8', 'ber_ma': 'ber_MA.UTF-8', 'bg': 'bg_BG.CP1251', 'bg_bg': 'bg_BG.CP1251', 'bhb_in.utf8': 'bhb_IN.UTF-8', 'bho_in': 'bho_IN.UTF-8', 'bho_np': 'bho_NP.UTF-8', 'bi_vu': 'bi_VU.UTF-8', 'bn_bd': 'bn_BD.UTF-8', 'bn_in': 'bn_IN.UTF-8', 'bo_cn': 'bo_CN.UTF-8', 'bo_in': 'bo_IN.UTF-8', 'bokmal': 'nb_NO.ISO8859-1', 'bokmål': 'nb_NO.ISO8859-1', 'br': 'br_FR.ISO8859-1', 'br_fr': 'br_FR.ISO8859-1', 'brx_in': 'brx_IN.UTF-8', 'bs': 'bs_BA.ISO8859-2', 'bs_ba': 'bs_BA.ISO8859-2', 'bulgarian': 'bg_BG.CP1251', 'byn_er': 'byn_ER.UTF-8', 'c': 'C', 'c-french': 'fr_CA.ISO8859-1', 'c.ascii': 'C', 'c.en': 'C', 'c.iso88591': 'en_US.ISO8859-1', 'c.utf8': 'en_US.UTF-8', 'c_c': 'C', 'c_c.c': 'C', 'ca': 'ca_ES.ISO8859-1', 'ca_ad': 'ca_AD.ISO8859-1', 'ca_es': 'ca_ES.ISO8859-1', 'ca_es@valencia': 'ca_ES.UTF-8@valencia', 'ca_fr': 'ca_FR.ISO8859-1', 'ca_it': 'ca_IT.ISO8859-1', 'catalan': 'ca_ES.ISO8859-1', 'ce_ru': 'ce_RU.UTF-8', 'cextend': 'en_US.ISO8859-1', 'chinese-s': 'zh_CN.eucCN', 'chinese-t': 'zh_TW.eucTW', 'chr_us': 'chr_US.UTF-8', 'ckb_iq': 'ckb_IQ.UTF-8', 'cmn_tw': 'cmn_TW.UTF-8', 'crh_ua': 'crh_UA.UTF-8', 'croatian': 'hr_HR.ISO8859-2', 'cs': 'cs_CZ.ISO8859-2', 'cs_cs': 'cs_CZ.ISO8859-2', 'cs_cz': 'cs_CZ.ISO8859-2', 'csb_pl': 'csb_PL.UTF-8', 'cv_ru': 'cv_RU.UTF-8', 'cy': 'cy_GB.ISO8859-1', 'cy_gb': 'cy_GB.ISO8859-1', 'cz': 'cs_CZ.ISO8859-2', 'cz_cz': 'cs_CZ.ISO8859-2', 'czech': 'cs_CZ.ISO8859-2', 'da': 'da_DK.ISO8859-1', 'da_dk': 'da_DK.ISO8859-1', 'danish': 'da_DK.ISO8859-1', 'dansk': 'da_DK.ISO8859-1', 'de': 'de_DE.ISO8859-1', 'de_at': 'de_AT.ISO8859-1', 'de_be': 'de_BE.ISO8859-1', 'de_ch': 'de_CH.ISO8859-1', 'de_de': 'de_DE.ISO8859-1', 'de_it': 'de_IT.ISO8859-1', 'de_li.utf8': 'de_LI.UTF-8', 'de_lu': 'de_LU.ISO8859-1', 'deutsch': 'de_DE.ISO8859-1', 'doi_in': 'doi_IN.UTF-8', 'dutch': 'nl_NL.ISO8859-1', 'dutch.iso88591': 'nl_BE.ISO8859-1', 'dv_mv': 'dv_MV.UTF-8', 'dz_bt': 'dz_BT.UTF-8', 'ee': 'ee_EE.ISO8859-4', 'ee_ee': 'ee_EE.ISO8859-4', 'eesti': 'et_EE.ISO8859-1', 'el': 'el_GR.ISO8859-7', 'el_cy': 'el_CY.ISO8859-7', 'el_gr': 'el_GR.ISO8859-7', 'el_gr@euro': 'el_GR.ISO8859-15', 'en': 'en_US.ISO8859-1', 'en_ag': 'en_AG.UTF-8', 'en_au': 'en_AU.ISO8859-1', 'en_be': 'en_BE.ISO8859-1', 'en_bw': 'en_BW.ISO8859-1', 'en_ca': 'en_CA.ISO8859-1', 'en_dk': 'en_DK.ISO8859-1', 'en_dl.utf8': 'en_DL.UTF-8', 'en_gb': 'en_GB.ISO8859-1', 'en_hk': 'en_HK.ISO8859-1', 'en_ie': 'en_IE.ISO8859-1', 'en_il': 'en_IL.UTF-8', 'en_in': 'en_IN.ISO8859-1', 'en_ng': 'en_NG.UTF-8', 'en_nz': 'en_NZ.ISO8859-1', 'en_ph': 'en_PH.ISO8859-1', 'en_sc.utf8': 'en_SC.UTF-8', 'en_sg': 'en_SG.ISO8859-1', 'en_uk': 'en_GB.ISO8859-1', 'en_us': 'en_US.ISO8859-1', 'en_us@euro@euro': 'en_US.ISO8859-15', 'en_za': 'en_ZA.ISO8859-1', 'en_zm': 'en_ZM.UTF-8', 'en_zw': 'en_ZW.ISO8859-1', 'en_zw.utf8': 'en_ZS.UTF-8', 'eng_gb': 'en_GB.ISO8859-1', 'english': 'en_EN.ISO8859-1', 'english.iso88591': 'en_US.ISO8859-1', 'english_uk': 'en_GB.ISO8859-1', 'english_united-states': 'en_US.ISO8859-1', 'english_united-states.437': 'C', 'english_us': 'en_US.ISO8859-1', 'eo': 'eo_XX.ISO8859-3', 'eo.utf8': 'eo.UTF-8', 'eo_eo': 'eo_EO.ISO8859-3', 'eo_us.utf8': 'eo_US.UTF-8', 'eo_xx': 'eo_XX.ISO8859-3', 'es': 'es_ES.ISO8859-1', 'es_ar': 'es_AR.ISO8859-1', 'es_bo': 'es_BO.ISO8859-1', 'es_cl': 'es_CL.ISO8859-1', 'es_co': 'es_CO.ISO8859-1', 'es_cr': 'es_CR.ISO8859-1', 'es_cu': 'es_CU.UTF-8', 'es_do': 'es_DO.ISO8859-1', 'es_ec': 'es_EC.ISO8859-1', 'es_es': 'es_ES.ISO8859-1', 'es_gt': 'es_GT.ISO8859-1', 'es_hn': 'es_HN.ISO8859-1', 'es_mx': 'es_MX.ISO8859-1', 'es_ni': 'es_NI.ISO8859-1', 'es_pa': 'es_PA.ISO8859-1', 'es_pe': 'es_PE.ISO8859-1', 'es_pr': 'es_PR.ISO8859-1', 'es_py': 'es_PY.ISO8859-1', 'es_sv': 'es_SV.ISO8859-1', 'es_us': 'es_US.ISO8859-1', 'es_uy': 'es_UY.ISO8859-1', 'es_ve': 'es_VE.ISO8859-1', 'estonian': 'et_EE.ISO8859-1', 'et': 'et_EE.ISO8859-15', 'et_ee': 'et_EE.ISO8859-15', 'eu': 'eu_ES.ISO8859-1', 'eu_es': 'eu_ES.ISO8859-1', 'eu_fr': 'eu_FR.ISO8859-1', 'fa': 'fa_IR.UTF-8', 'fa_ir': 'fa_IR.UTF-8', 'fa_ir.isiri3342': 'fa_IR.ISIRI-3342', 'ff_sn': 'ff_SN.UTF-8', 'fi': 'fi_FI.ISO8859-15', 'fi_fi': 'fi_FI.ISO8859-15', 'fil_ph': 'fil_PH.UTF-8', 'finnish': 'fi_FI.ISO8859-1', 'fo': 'fo_FO.ISO8859-1', 'fo_fo': 'fo_FO.ISO8859-1', 'fr': 'fr_FR.ISO8859-1', 'fr_be': 'fr_BE.ISO8859-1', 'fr_ca': 'fr_CA.ISO8859-1', 'fr_ch': 'fr_CH.ISO8859-1', 'fr_fr': 'fr_FR.ISO8859-1', 'fr_lu': 'fr_LU.ISO8859-1', 'français': 'fr_FR.ISO8859-1', 'fre_fr': 'fr_FR.ISO8859-1', 'french': 'fr_FR.ISO8859-1', 'french.iso88591': 'fr_CH.ISO8859-1', 'french_france': 'fr_FR.ISO8859-1', 'fur_it': 'fur_IT.UTF-8', 'fy_de': 'fy_DE.UTF-8', 'fy_nl': 'fy_NL.UTF-8', 'ga': 'ga_IE.ISO8859-1', 'ga_ie': 'ga_IE.ISO8859-1', 'galego': 'gl_ES.ISO8859-1', 'galician': 'gl_ES.ISO8859-1', 'gd': 'gd_GB.ISO8859-1', 'gd_gb': 'gd_GB.ISO8859-1', 'ger_de': 'de_DE.ISO8859-1', 'german': 'de_DE.ISO8859-1', 'german.iso88591': 'de_CH.ISO8859-1', 'german_germany': 'de_DE.ISO8859-1', 'gez_er': 'gez_ER.UTF-8', 'gez_et': 'gez_ET.UTF-8', 'gl': 'gl_ES.ISO8859-1', 'gl_es': 'gl_ES.ISO8859-1', 'greek': 'el_GR.ISO8859-7', 'gu_in': 'gu_IN.UTF-8', 'gv': 'gv_GB.ISO8859-1', 'gv_gb': 'gv_GB.ISO8859-1', 'ha_ng': 'ha_NG.UTF-8', 'hak_tw': 'hak_TW.UTF-8', 'he': 'he_IL.ISO8859-8', 'he_il': 'he_IL.ISO8859-8', 'hebrew': 'he_IL.ISO8859-8', 'hi': 'hi_IN.ISCII-DEV', 'hi_in': 'hi_IN.ISCII-DEV', 'hi_in.isciidev': 'hi_IN.ISCII-DEV', 'hif_fj': 'hif_FJ.UTF-8', 'hne': 'hne_IN.UTF-8', 'hne_in': 'hne_IN.UTF-8', 'hr': 'hr_HR.ISO8859-2', 'hr_hr': 'hr_HR.ISO8859-2', 'hrvatski': 'hr_HR.ISO8859-2', 'hsb_de': 'hsb_DE.ISO8859-2', 'ht_ht': 'ht_HT.UTF-8', 'hu': 'hu_HU.ISO8859-2', 'hu_hu': 'hu_HU.ISO8859-2', 'hungarian': 'hu_HU.ISO8859-2', 'hy_am': 'hy_AM.UTF-8', 'hy_am.armscii8': 'hy_AM.ARMSCII_8', 'ia': 'ia.UTF-8', 'ia_fr': 'ia_FR.UTF-8', 'icelandic': 'is_IS.ISO8859-1', 'id': 'id_ID.ISO8859-1', 'id_id': 'id_ID.ISO8859-1', 'ig_ng': 'ig_NG.UTF-8', 'ik_ca': 'ik_CA.UTF-8', 'in': 'id_ID.ISO8859-1', 'in_id': 'id_ID.ISO8859-1', 'is': 'is_IS.ISO8859-1', 'is_is': 'is_IS.ISO8859-1', 'iso-8859-1': 'en_US.ISO8859-1', 'iso-8859-15': 'en_US.ISO8859-15', 'iso8859-1': 'en_US.ISO8859-1', 'iso8859-15': 'en_US.ISO8859-15', 'iso_8859_1': 'en_US.ISO8859-1', 'iso_8859_15': 'en_US.ISO8859-15', 'it': 'it_IT.ISO8859-1', 'it_ch': 'it_CH.ISO8859-1', 'it_it': 'it_IT.ISO8859-1', 'italian': 'it_IT.ISO8859-1', 'iu': 'iu_CA.NUNACOM-8', 'iu_ca': 'iu_CA.NUNACOM-8', 'iu_ca.nunacom8': 'iu_CA.NUNACOM-8', 'iw': 'he_IL.ISO8859-8', 'iw_il': 'he_IL.ISO8859-8', 'iw_il.utf8': 'iw_IL.UTF-8', 'ja': 'ja_JP.eucJP', 'ja_jp': 'ja_JP.eucJP', 'ja_jp.euc': 'ja_JP.eucJP', 'ja_jp.mscode': 'ja_JP.SJIS', 'ja_jp.pck': 'ja_JP.SJIS', 'japan': 'ja_JP.eucJP', 'japanese': 'ja_JP.eucJP', 'japanese-euc': 'ja_JP.eucJP', 'japanese.euc': 'ja_JP.eucJP', 'jp_jp': 'ja_JP.eucJP', 'ka': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge.georgianacademy': 'ka_GE.GEORGIAN-ACADEMY', 'ka_ge.georgianps': 'ka_GE.GEORGIAN-PS', 'ka_ge.georgianrs': 'ka_GE.GEORGIAN-ACADEMY', 'kab_dz': 'kab_DZ.UTF-8', 'kk_kz': 'kk_KZ.ptcp154', 'kl': 'kl_GL.ISO8859-1', 'kl_gl': 'kl_GL.ISO8859-1', 'km_kh': 'km_KH.UTF-8', 'kn': 'kn_IN.UTF-8', 'kn_in': 'kn_IN.UTF-8', 'ko': 'ko_KR.eucKR', 'ko_kr': 'ko_KR.eucKR', 'ko_kr.euc': 'ko_KR.eucKR', 'kok_in': 'kok_IN.UTF-8', 'korean': 'ko_KR.eucKR', 'korean.euc': 'ko_KR.eucKR', 'ks': 'ks_IN.UTF-8', 'ks_in': 'ks_IN.UTF-8', 'ks_in@devanagari.utf8': 'ks_IN.UTF-8@devanagari', 'ku_tr': 'ku_TR.ISO8859-9', 'kw': 'kw_GB.ISO8859-1', 'kw_gb': 'kw_GB.ISO8859-1', 'ky': 'ky_KG.UTF-8', 'ky_kg': 'ky_KG.UTF-8', 'lb_lu': 'lb_LU.UTF-8', 'lg_ug': 'lg_UG.ISO8859-10', 'li_be': 'li_BE.UTF-8', 'li_nl': 'li_NL.UTF-8', 'lij_it': 'lij_IT.UTF-8', 'lithuanian': 'lt_LT.ISO8859-13', 'ln_cd': 'ln_CD.UTF-8', 'lo': 'lo_LA.MULELAO-1', 'lo_la': 'lo_LA.MULELAO-1', 'lo_la.cp1133': 'lo_LA.IBM-CP1133', 'lo_la.ibmcp1133': 'lo_LA.IBM-CP1133', 'lo_la.mulelao1': 'lo_LA.MULELAO-1', 'lt': 'lt_LT.ISO8859-13', 'lt_lt': 'lt_LT.ISO8859-13', 'lv': 'lv_LV.ISO8859-13', 'lv_lv': 'lv_LV.ISO8859-13', 'lzh_tw': 'lzh_TW.UTF-8', 'mag_in': 'mag_IN.UTF-8', 'mai': 'mai_IN.UTF-8', 'mai_in': 'mai_IN.UTF-8', 'mai_np': 'mai_NP.UTF-8', 'mfe_mu': 'mfe_MU.UTF-8', 'mg_mg': 'mg_MG.ISO8859-15', 'mhr_ru': 'mhr_RU.UTF-8', 'mi': 'mi_NZ.ISO8859-1', 'mi_nz': 'mi_NZ.ISO8859-1', 'miq_ni': 'miq_NI.UTF-8', 'mjw_in': 'mjw_IN.UTF-8', 'mk': 'mk_MK.ISO8859-5', 'mk_mk': 'mk_MK.ISO8859-5', 'ml': 'ml_IN.UTF-8', 'ml_in': 'ml_IN.UTF-8', 'mn_mn': 'mn_MN.UTF-8', 'mni_in': 'mni_IN.UTF-8', 'mr': 'mr_IN.UTF-8', 'mr_in': 'mr_IN.UTF-8', 'ms': 'ms_MY.ISO8859-1', 'ms_my': 'ms_MY.ISO8859-1', 'mt': 'mt_MT.ISO8859-3', 'mt_mt': 'mt_MT.ISO8859-3', 'my_mm': 'my_MM.UTF-8', 'nan_tw': 'nan_TW.UTF-8', 'nb': 'nb_NO.ISO8859-1', 'nb_no': 'nb_NO.ISO8859-1', 'nds_de': 'nds_DE.UTF-8', 'nds_nl': 'nds_NL.UTF-8', 'ne_np': 'ne_NP.UTF-8', 'nhn_mx': 'nhn_MX.UTF-8', 'niu_nu': 'niu_NU.UTF-8', 'niu_nz': 'niu_NZ.UTF-8', 'nl': 'nl_NL.ISO8859-1', 'nl_aw': 'nl_AW.UTF-8', 'nl_be': 'nl_BE.ISO8859-1', 'nl_nl': 'nl_NL.ISO8859-1', 'nn': 'nn_NO.ISO8859-1', 'nn_no': 'nn_NO.ISO8859-1', 'no': 'no_NO.ISO8859-1', 'no@nynorsk': 'ny_NO.ISO8859-1', 'no_no': 'no_NO.ISO8859-1', 'no_no.iso88591@bokmal': 'no_NO.ISO8859-1', 'no_no.iso88591@nynorsk': 'no_NO.ISO8859-1', 'norwegian': 'no_NO.ISO8859-1', 'nr': 'nr_ZA.ISO8859-1', 'nr_za': 'nr_ZA.ISO8859-1', 'nso': 'nso_ZA.ISO8859-15', 'nso_za': 'nso_ZA.ISO8859-15', 'ny': 'ny_NO.ISO8859-1', 'ny_no': 'ny_NO.ISO8859-1', 'nynorsk': 'nn_NO.ISO8859-1', 'oc': 'oc_FR.ISO8859-1', 'oc_fr': 'oc_FR.ISO8859-1', 'om_et': 'om_ET.UTF-8', 'om_ke': 'om_KE.ISO8859-1', 'or': 'or_IN.UTF-8', 'or_in': 'or_IN.UTF-8', 'os_ru': 'os_RU.UTF-8', 'pa': 'pa_IN.UTF-8', 'pa_in': 'pa_IN.UTF-8', 'pa_pk': 'pa_PK.UTF-8', 'pap_an': 'pap_AN.UTF-8', 'pap_aw': 'pap_AW.UTF-8', 'pap_cw': 'pap_CW.UTF-8', 'pd': 'pd_US.ISO8859-1', 'pd_de': 'pd_DE.ISO8859-1', 'pd_us': 'pd_US.ISO8859-1', 'ph': 'ph_PH.ISO8859-1', 'ph_ph': 'ph_PH.ISO8859-1', 'pl': 'pl_PL.ISO8859-2', 'pl_pl': 'pl_PL.ISO8859-2', 'polish': 'pl_PL.ISO8859-2', 'portuguese': 'pt_PT.ISO8859-1', 'portuguese_brazil': 'pt_BR.ISO8859-1', 'posix': 'C', 'posix-utf2': 'C', 'pp': 'pp_AN.ISO8859-1', 'pp_an': 'pp_AN.ISO8859-1', 'ps_af': 'ps_AF.UTF-8', 'pt': 'pt_PT.ISO8859-1', 'pt_br': 'pt_BR.ISO8859-1', 'pt_pt': 'pt_PT.ISO8859-1', 'quz_pe': 'quz_PE.UTF-8', 'raj_in': 'raj_IN.UTF-8', 'ro': 'ro_RO.ISO8859-2', 'ro_ro': 'ro_RO.ISO8859-2', 'romanian': 'ro_RO.ISO8859-2', 'ru': 'ru_RU.UTF-8', 'ru_ru': 'ru_RU.UTF-8', 'ru_ua': 'ru_UA.KOI8-U', 'rumanian': 'ro_RO.ISO8859-2', 'russian': 'ru_RU.KOI8-R', 'rw': 'rw_RW.ISO8859-1', 'rw_rw': 'rw_RW.ISO8859-1', 'sa_in': 'sa_IN.UTF-8', 'sat_in': 'sat_IN.UTF-8', 'sc_it': 'sc_IT.UTF-8', 'sd': 'sd_IN.UTF-8', 'sd_in': 'sd_IN.UTF-8', 'sd_in@devanagari.utf8': 'sd_IN.UTF-8@devanagari', 'sd_pk': 'sd_PK.UTF-8', 'se_no': 'se_NO.UTF-8', 'serbocroatian': 'sr_RS.UTF-8@latin', 'sgs_lt': 'sgs_LT.UTF-8', 'sh': 'sr_RS.UTF-8@latin', 'sh_ba.iso88592@bosnia': 'sr_CS.ISO8859-2', 'sh_hr': 'sh_HR.ISO8859-2', 'sh_hr.iso88592': 'hr_HR.ISO8859-2', 'sh_sp': 'sr_CS.ISO8859-2', 'sh_yu': 'sr_RS.UTF-8@latin', 'shn_mm': 'shn_MM.UTF-8', 'shs_ca': 'shs_CA.UTF-8', 'si': 'si_LK.UTF-8', 'si_lk': 'si_LK.UTF-8', 'sid_et': 'sid_ET.UTF-8', 'sinhala': 'si_LK.UTF-8', 'sk': 'sk_SK.ISO8859-2', 'sk_sk': 'sk_SK.ISO8859-2', 'sl': 'sl_SI.ISO8859-2', 'sl_cs': 'sl_CS.ISO8859-2', 'sl_si': 'sl_SI.ISO8859-2', 'slovak': 'sk_SK.ISO8859-2', 'slovene': 'sl_SI.ISO8859-2', 'slovenian': 'sl_SI.ISO8859-2', 'sm_ws': 'sm_WS.UTF-8', 'so_dj': 'so_DJ.ISO8859-1', 'so_et': 'so_ET.UTF-8', 'so_ke': 'so_KE.ISO8859-1', 'so_so': 'so_SO.ISO8859-1', 'sp': 'sr_CS.ISO8859-5', 'sp_yu': 'sr_CS.ISO8859-5', 'spanish': 'es_ES.ISO8859-1', 'spanish_spain': 'es_ES.ISO8859-1', 'sq': 'sq_AL.ISO8859-2', 'sq_al': 'sq_AL.ISO8859-2', 'sq_mk': 'sq_MK.UTF-8', 'sr': 'sr_RS.UTF-8', 'sr@cyrillic': 'sr_RS.UTF-8', 'sr@latn': 'sr_CS.UTF-8@latin', 'sr_cs': 'sr_CS.UTF-8', 'sr_cs.iso88592@latn': 'sr_CS.ISO8859-2', 'sr_cs@latn': 'sr_CS.UTF-8@latin', 'sr_me': 'sr_ME.UTF-8', 'sr_rs': 'sr_RS.UTF-8', 'sr_rs@latn': 'sr_RS.UTF-8@latin', 'sr_sp': 'sr_CS.ISO8859-2', 'sr_yu': 'sr_RS.UTF-8@latin', 'sr_yu.cp1251@cyrillic': 'sr_CS.CP1251', 'sr_yu.iso88592': 'sr_CS.ISO8859-2', 'sr_yu.iso88595': 'sr_CS.ISO8859-5', 'sr_yu.iso88595@cyrillic': 'sr_CS.ISO8859-5', 'sr_yu.microsoftcp1251@cyrillic': 'sr_CS.CP1251', 'sr_yu.utf8': 'sr_RS.UTF-8', 'sr_yu.utf8@cyrillic': 'sr_RS.UTF-8', 'sr_yu@cyrillic': 'sr_RS.UTF-8', 'ss': 'ss_ZA.ISO8859-1', 'ss_za': 'ss_ZA.ISO8859-1', 'st': 'st_ZA.ISO8859-1', 'st_za': 'st_ZA.ISO8859-1', 'sv': 'sv_SE.ISO8859-1', 'sv_fi': 'sv_FI.ISO8859-1', 'sv_se': 'sv_SE.ISO8859-1', 'sw_ke': 'sw_KE.UTF-8', 'sw_tz': 'sw_TZ.UTF-8', 'swedish': 'sv_SE.ISO8859-1', 'szl_pl': 'szl_PL.UTF-8', 'ta': 'ta_IN.TSCII-0', 'ta_in': 'ta_IN.TSCII-0', 'ta_in.tscii': 'ta_IN.TSCII-0', 'ta_in.tscii0': 'ta_IN.TSCII-0', 'ta_lk': 'ta_LK.UTF-8', 'tcy_in.utf8': 'tcy_IN.UTF-8', 'te': 'te_IN.UTF-8', 'te_in': 'te_IN.UTF-8', 'tg': 'tg_TJ.KOI8-C', 'tg_tj': 'tg_TJ.KOI8-C', 'th': 'th_TH.ISO8859-11', 'th_th': 'th_TH.ISO8859-11', 'th_th.tactis': 'th_TH.TIS620', 'th_th.tis620': 'th_TH.TIS620', 'thai': 'th_TH.ISO8859-11', 'the_np': 'the_NP.UTF-8', 'ti_er': 'ti_ER.UTF-8', 'ti_et': 'ti_ET.UTF-8', 'tig_er': 'tig_ER.UTF-8', 'tk_tm': 'tk_TM.UTF-8', 'tl': 'tl_PH.ISO8859-1', 'tl_ph': 'tl_PH.ISO8859-1', 'tn': 'tn_ZA.ISO8859-15', 'tn_za': 'tn_ZA.ISO8859-15', 'to_to': 'to_TO.UTF-8', 'tpi_pg': 'tpi_PG.UTF-8', 'tr': 'tr_TR.ISO8859-9', 'tr_cy': 'tr_CY.ISO8859-9', 'tr_tr': 'tr_TR.ISO8859-9', 'ts': 'ts_ZA.ISO8859-1', 'ts_za': 'ts_ZA.ISO8859-1', 'tt': 'tt_RU.TATAR-CYR', 'tt_ru': 'tt_RU.TATAR-CYR', 'tt_ru.tatarcyr': 'tt_RU.TATAR-CYR', 'tt_ru@iqtelif': 'tt_RU.UTF-8@iqtelif', 'turkish': 'tr_TR.ISO8859-9', 'ug_cn': 'ug_CN.UTF-8', 'uk': 'uk_UA.KOI8-U', 'uk_ua': 'uk_UA.KOI8-U', 'univ': 'en_US.utf', 'universal': 'en_US.utf', 'universal.utf8@ucs4': 'en_US.UTF-8', 'unm_us': 'unm_US.UTF-8', 'ur': 'ur_PK.CP1256', 'ur_in': 'ur_IN.UTF-8', 'ur_pk': 'ur_PK.CP1256', 'uz': 'uz_UZ.UTF-8', 'uz_uz': 'uz_UZ.UTF-8', 'uz_uz@cyrillic': 'uz_UZ.UTF-8', 've': 've_ZA.UTF-8', 've_za': 've_ZA.UTF-8', 'vi': 'vi_VN.TCVN', 'vi_vn': 'vi_VN.TCVN', 'vi_vn.tcvn': 'vi_VN.TCVN', 'vi_vn.tcvn5712': 'vi_VN.TCVN', 'vi_vn.viscii': 'vi_VN.VISCII', 'vi_vn.viscii111': 'vi_VN.VISCII', 'wa': 'wa_BE.ISO8859-1', 'wa_be': 'wa_BE.ISO8859-1', 'wae_ch': 'wae_CH.UTF-8', 'wal_et': 'wal_ET.UTF-8', 'wo_sn': 'wo_SN.UTF-8', 'xh': 'xh_ZA.ISO8859-1', 'xh_za': 'xh_ZA.ISO8859-1', 'yi': 'yi_US.CP1255', 'yi_us': 'yi_US.CP1255', 'yo_ng': 'yo_NG.UTF-8', 'yue_hk': 'yue_HK.UTF-8', 'yuw_pg': 'yuw_PG.UTF-8', 'zh': 'zh_CN.eucCN', 'zh_cn': 'zh_CN.gb2312', 'zh_cn.big5': 'zh_TW.big5', 'zh_cn.euc': 'zh_CN.eucCN', 'zh_hk': 'zh_HK.big5hkscs', 'zh_hk.big5hk': 'zh_HK.big5hkscs', 'zh_sg': 'zh_SG.GB2312', 'zh_sg.gbk': 'zh_SG.GBK', 'zh_tw': 'zh_TW.big5', 'zh_tw.euc': 'zh_TW.eucTW', 'zh_tw.euctw': 'zh_TW.eucTW', 'zu': 'zu_ZA.ISO8859-1', 'zu_za': 'zu_ZA.ISO8859-1'}
    assert module_1.windows_locale == {1078: 'af_ZA', 1052: 'sq_AL', 1156: 'gsw_FR', 1118: 'am_ET', 1025: 'ar_SA', 2049: 'ar_IQ', 3073: 'ar_EG', 4097: 'ar_LY', 5121: 'ar_DZ', 6145: 'ar_MA', 7169: 'ar_TN', 8193: 'ar_OM', 9217: 'ar_YE', 10241: 'ar_SY', 11265: 'ar_JO', 12289: 'ar_LB', 13313: 'ar_KW', 14337: 'ar_AE', 15361: 'ar_BH', 16385: 'ar_QA', 1067: 'hy_AM', 1101: 'as_IN', 1068: 'az_AZ', 2092: 'az_AZ', 1133: 'ba_RU', 1069: 'eu_ES', 1059: 'be_BY', 1093: 'bn_IN', 8218: 'bs_BA', 5146: 'bs_BA', 1150: 'br_FR', 1026: 'bg_BG', 1027: 'ca_ES', 4: 'zh_CHS', 1028: 'zh_TW', 2052: 'zh_CN', 3076: 'zh_HK', 4100: 'zh_SG', 5124: 'zh_MO', 31748: 'zh_CHT', 1155: 'co_FR', 1050: 'hr_HR', 4122: 'hr_BA', 1029: 'cs_CZ', 1030: 'da_DK', 1164: 'gbz_AF', 1125: 'div_MV', 1043: 'nl_NL', 2067: 'nl_BE', 1033: 'en_US', 2057: 'en_GB', 3081: 'en_AU', 4105: 'en_CA', 5129: 'en_NZ', 6153: 'en_IE', 7177: 'en_ZA', 8201: 'en_JA', 9225: 'en_CB', 10249: 'en_BZ', 11273: 'en_TT', 12297: 'en_ZW', 13321: 'en_PH', 16393: 'en_IN', 17417: 'en_MY', 18441: 'en_IN', 1061: 'et_EE', 1080: 'fo_FO', 1124: 'fil_PH', 1035: 'fi_FI', 1036: 'fr_FR', 2060: 'fr_BE', 3084: 'fr_CA', 4108: 'fr_CH', 5132: 'fr_LU', 6156: 'fr_MC', 1122: 'fy_NL', 1110: 'gl_ES', 1079: 'ka_GE', 1031: 'de_DE', 2055: 'de_CH', 3079: 'de_AT', 4103: 'de_LU', 5127: 'de_LI', 1032: 'el_GR', 1135: 'kl_GL', 1095: 'gu_IN', 1128: 'ha_NG', 1037: 'he_IL', 1081: 'hi_IN', 1038: 'hu_HU', 1039: 'is_IS', 1057: 'id_ID', 1117: 'iu_CA', 2141: 'iu_CA', 2108: 'ga_IE', 1040: 'it_IT', 2064: 'it_CH', 1041: 'ja_JP', 1099: 'kn_IN', 1087: 'kk_KZ', 1107: 'kh_KH', 1158: 'qut_GT', 1159: 'rw_RW', 1111: 'kok_IN', 1042: 'ko_KR', 1088: 'ky_KG', 1108: 'lo_LA', 1062: 'lv_LV', 1063: 'lt_LT', 2094: 'dsb_DE', 1134: 'lb_LU', 1071: 'mk_MK', 1086: 'ms_MY', 2110: 'ms_BN', 1100: 'ml_IN', 1082: 'mt_MT', 1153: 'mi_NZ', 1146: 'arn_CL', 1102: 'mr_IN', 1148: 'moh_CA', 1104: 'mn_MN', 2128: 'mn_CN', 1121: 'ne_NP', 1044: 'nb_NO', 2068: 'nn_NO', 1154: 'oc_FR', 1096: 'or_IN', 1123: 'ps_AF', 1065: 'fa_IR', 1045: 'pl_PL', 1046: 'pt_BR', 2070: 'pt_PT', 1094: 'pa_IN', 1131: 'quz_BO', 2155: 'quz_EC', 3179: 'quz_PE', 1048: 'ro_RO', 1047: 'rm_CH', 1049: 'ru_RU', 9275: 'smn_FI', 4155: 'smj_NO', 5179: 'smj_SE', 1083: 'se_NO', 2107: 'se_SE', 3131: 'se_FI', 8251: 'sms_FI', 6203: 'sma_NO', 7227: 'sma_SE', 1103: 'sa_IN', 3098: 'sr_SP', 7194: 'sr_BA', 2074: 'sr_SP', 6170: 'sr_BA', 1115: 'si_LK', 1132: 'ns_ZA', 1074: 'tn_ZA', 1051: 'sk_SK', 1060: 'sl_SI', 1034: 'es_ES', 2058: 'es_MX', 3082: 'es_ES', 4106: 'es_GT', 5130: 'es_CR', 6154: 'es_PA', 7178: 'es_DO', 8202: 'es_VE', 9226: 'es_CO', 10250: 'es_PE', 11274: 'es_AR', 12298: 'es_EC', 13322: 'es_CL', 14346: 'es_UR', 15370: 'es_PY', 16394: 'es_BO', 17418: 'es_SV', 18442: 'es_HN', 19466: 'es_NI', 20490: 'es_PR', 21514: 'es_US', 1089: 'sw_KE', 1053: 'sv_SE', 2077: 'sv_FI', 1114: 'syr_SY', 1064: 'tg_TJ', 2143: 'tmz_DZ', 1097: 'ta_IN', 1092: 'tt_RU', 1098: 'te_IN', 1054: 'th_TH', 2129: 'bo_BT', 1105: 'bo_CN', 1055: 'tr_TR', 1090: 'tk_TM', 1152: 'ug_CN', 1058: 'uk_UA', 1070: 'wen_DE', 1056: 'ur_PK', 2080: 'ur_IN', 1091: 'uz_UZ', 2115: 'uz_UZ', 1066: 'vi_VN', 1106: 'cy_GB', 1160: 'wo_SN', 1076: 'xh_ZA', 1157: 'sah_RU', 1144: 'ii_CN', 1130: 'yo_NG', 1077: 'zu_ZA'}
    var_1 = var_0.__repr__()
    assert var_1 == "'UTF-8'"
    module_0.process(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = module_4.Morsel()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'http.cookies.Morsel'
    assert len(var_0) == 9
    assert f'{type(module_4.Morsel.key).__module__}.{type(module_4.Morsel.key).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Morsel.value).__module__}.{type(module_4.Morsel.value).__qualname__}' == 'builtins.property'
    assert f'{type(module_4.Morsel.coded_value).__module__}.{type(module_4.Morsel.coded_value).__qualname__}' == 'builtins.property'
    var_1 = var_0.js_output()
    assert var_1 == '\n        <script type="text/javascript">\n        <!-- begin hiding\n        document.cookie = "None=None";\n        // end hiding -->\n        </script>\n        '
    module_0.process(var_1, var_1, var_1, var_1)