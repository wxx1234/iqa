c1 = 0.46
c2 = 0.54
c3 = 0.54
c4 = 0.48
c5 = 0.52
c6 = 0.52


def get_mos_live_session(qs, qi_live_session, qv_live_session):
    mos_live_session = (qs - 1) * (1 - c1 * (5 - qi_live_session) - c2 * (5 - qv_live_session)) + 1
    return mos_live_session


def get_mos_live_instant(qs, qv_live_instant):
    mos_live_instant = (qs - 1) * (1 - c3 * (5 - qv_live_instant)) + 1
    return mos_live_instant


def get_mos_vod_session(qs, qi_vod_session, qv_vod_session):
    mos_vod_session = (qs - 1) * (1 - c4 * (5 - qi_vod_session) + c5 * (5 - qv_vod_session)) + 1
    return mos_vod_session


def get_mos_vod_instant(qs, qv_vod_instant):
    mos_vod_instant = (qs - 1) * (1 - c6 * (5 - qv_vod_instant)) + 1
    return mos_vod_instant
