import math

c15 = 2.249
c16 = -0.001127
c17 = 3.09
c18 = -0.0002869
alpha_1 = 2.249
beta_1 = -0.001127
alpha_2 = 3.09
beta_2 = -0.0002869
c19 = -0.00000000001689
c20 = 0.0000002583
c21 = -0.001508
c22 = 5.211
alpha_3 = 0.5852
beta_3 = -3.641
alpha_4 = 0.4139
beta_4 = -0.009638


def quality_of_live_session(t_zapping, t, T=0):
    q_zapping = c15 * math.exp(c16 * t_zapping) + c17 * math.exp(c18 * t_zapping)
    qi_live_instant = q_zapping
    if T:
        qi_live_session = (alpha_1 * math.exp(beta_1 * t / T) + alpha_2 * math.exp(beta_2 * t / T)) * qi_live_instant
        return qi_live_session
    else:
        return qi_live_instant


def quality_of_vod_session(t_loading, t, T=0):
    q_loading = c19 * (t_loading ** 3) + c20 * (t_loading ** 2) + c21 * t_loading + c22
    qi_vod_instant = q_loading
    if T:
        qi_vod_session = (alpha_3 * math.exp(beta_3 * t / T) + alpha_4 * math.exp(beta_4 * t / T)) * qi_vod_instant
        return qi_vod_session
    else:
        return qi_vod_instant
