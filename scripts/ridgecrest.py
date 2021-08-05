import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlb
import matplotlib.cm as cm
from faultSlip.inversion import Inversion


def dens2sparss_missfit(inv, in_beta, min, max, num=30):
    alphas = np.logspace(min, max, num)
    spars_misfit = []
    dens_misfit = []

    for alpha in alphas:
        print(alpha)
        inv.solve_CA(in_beta, alpha)

        mask1 = np.concatenate((inv.gps[0].data.E, inv.gps[0].data.N, inv.gps[0].data.Up)) != 0
        b_1 = inv.gps[0].get_data()
        G_1 = np.concatenate((inv.gps[0].G_ss, inv.gps[0].G_ds), axis=1)
        G_1 = G_1[mask1]
        b_1 = b_1[mask1]
        w_1 = inv.gps[0].cala_whigts(mask1)

        mask2 = np.concatenate((inv.gps[1].data.E, inv.gps[1].data.N, inv.gps[1].data.Up)) != 0
        b_2 = inv.gps[1].get_data()
        G_2 = np.concatenate((inv.gps[1].G_ss, inv.gps[1].G_ds), axis=1)
        G_2 = G_2[mask2]
        b_2 = b_2[mask2]
        w_2 = inv.gps[1].cala_whigts(mask2)

        G_1t = np.concatenate((G_1, np.zeros((G_1.shape[0], G_2.shape[1]))), axis=1)
        G_2t = np.concatenate((np.zeros((G_2.shape[0], G_1.shape[1])), G_2), axis=1)

        w_spars = np.concatenate((w_1, w_2))
        G_spars = np.concatenate((G_1t, G_2t)) * w_spars.reshape(-1, 1)
        G_spars = np.concatenate((G_spars, np.zeros((G_spars.shape[0], 2))), axis=1)
        b_spars = np.concatenate((b_1, b_2)) * w_spars

        b_sar1, G_sar1 = inv.images[0].get_image_kersNdata()
        b_sar2, G_sar2 = inv.images[0].get_image_kersNdata()
        G_sar = np.concatenate((G_sar1, G_sar2), axis=0)
        G_sar = np.concatenate((G_sar, G_sar), axis=1)
        b_sar = np.concatenate((b_sar1, b_sar2))
        #
        offset_1 = np.concatenate(
            (np.ones(G_sar1.shape[0]).reshape(-1, 1) * 1, np.zeros(G_sar1.shape[0]).reshape(-1, 1)), axis=1)
        offset_2 = np.concatenate(
            (np.zeros(G_sar2.shape[0]).reshape(-1, 1), np.ones(G_sar2.shape[0]).reshape(-1, 1) * 1),
            axis=1)
        offset = np.concatenate((offset_1, offset_2), axis=0)
        G_sar = np.concatenate((G_sar, offset), axis=1)

        m = inv.solution
        ks = np.array([len(p.sources) for p in inv.gps[0].plains])
        k_1 = np.sum(ks[:-1])
        k_2 = ks[-1]
        mask_rl = np.concatenate((
                                 np.ones(k_1, dtype=np.bool), np.zeros(k_2, dtype=np.bool), np.ones(k_1, dtype=np.bool),
                                 np.zeros(k_2, dtype=np.bool), np.ones(k_1, dtype=np.bool),
                                 np.zeros(k_2, dtype=np.bool), np.ones(k_1, dtype=np.bool),
                                 np.zeros(k_2, dtype=np.bool)))

        spars_misfit.append(np.linalg.norm(G_spars.dot(m) - b_spars) / np.linalg.norm(b_spars))
        # spars_misfit.append(np.sqrt(np.linalg.norm(G_spars.dot(m) - b_spars, 1) / b_spars.shape[0]))
        # m[mask_rl] *= -1
        dens_misfit.append(np.linalg.norm(G_sar.dot(m) - b_sar) / np.linalg.norm(b_sar))
        # dens_misfit.append(np.sqrt(np.linalg.norm(G_sar.dot(m) - b_sar, 1) / b_sar.shape[0]))
    return spars_misfit, dens_misfit


def calculate_missfit(inv, im2sar):
    mask1 = np.concatenate((inv.gps[0].data.E, inv.gps[0].data.N, inv.gps[0].data.Up)) != 0
    b_1 = inv.gps[0].get_data()
    G_1 = np.concatenate((inv.gps[0].G_ss, inv.gps[0].G_ds), axis=1)
    G_1 = G_1[mask1]
    b_1 = b_1[mask1]
    w_1 = inv.gps[0].cala_whigts(mask1)

    mask2 = np.concatenate((inv.gps[1].data.E, inv.gps[1].data.N, inv.gps[1].data.Up)) != 0
    b_2 = inv.gps[1].get_data()
    G_2 = np.concatenate((inv.gps[1].G_ss, inv.gps[1].G_ds), axis=1)
    G_2 = G_2[mask2]
    b_2 = b_2[mask2]
    w_2 = inv.gps[1].cala_whigts(mask2)

    # mask3 = np.concatenate((inv.gps[2].data.E, inv.gps[2].data.N, inv.gps[2].data.Up)) != 0
    # b_3 = inv.gps[2].get_data()
    # w_3 = inv.gps[2].cala_whigts(mask3)
    # G_3 = np.concatenate((inv.gps[2].G_ss, inv.gps[2].G_ds), axis=1)
    # G_3 = G_3[mask3]
    # b_3 = b_3[mask3]

    G_1t = np.concatenate((G_1, np.zeros((G_1.shape[0], G_2.shape[1]))), axis=1)
    G_2t = np.concatenate((np.zeros((G_2.shape[0], G_1.shape[1])), G_2), axis=1)
    # G_3t = np.concatenate((np.zeros((G_3.shape[0], G_1.shape[1])), G_3), axis=1)

    G_gps = np.concatenate((G_1t, G_2t)) * np.concatenate((w_1, w_2)).reshape(-1, 1)
    b_gps = np.concatenate((b_1, b_2)) * np.concatenate((w_1, w_2))

    # G_profile = G_3t * w_3.reshape(-1,1)
    # b_3 = b_3 * w_3

    G_spars = G_gps  # np.concatenate((G_gps, G_profile))
    b_spars = b_gps  # np.concatenate((b_gps, b_3))

    b_spot, G_spot = inv.get_sar_inv_pars(False, (2, 4))
    b_spot = b_spot / im2sar
    G_spot = np.concatenate((G_spot, np.zeros_like(G_spot)), axis=1) / im2sar

    b_sar, G_sar = inv.get_sar_inv_pars(False, (0, 2))
    G_sar = np.concatenate((G_sar, G_sar), axis=1)

    G_dens = np.concatenate((G_sar, G_spot), axis=0)
    b_dens = np.concatenate((b_sar, b_spot))

    m = inv.solution
    ks = np.array([len(p.sources) for p in inv.gps[0].plains])

    x = []
    inv.assign_slip()
    for i in (19, 20, 21):
        for s in inv.plains[i].sources:
            if s.depth_t < 0.5:
                x.append(s.strike_slip)

    x = np.array(x)
    y = np.array(
        [0.9226381956589431, 1.2225717588576088, 0.8985328915287425, 0.6385295812081863, 0.6215600130290337,
         0.6428820928777036, 0.7375838786029231])

    spars_misfit = np.linalg.norm(G_spars.dot(m) - b_spars) / np.linalg.norm(b_spars)
    # m[mask_rl] *= -1
    dens_misfit = np.linalg.norm(G_dens.dot(m) - b_dens) / np.linalg.norm(b_dens)
    sar_misfit = np.linalg.norm(G_sar.dot(m) - b_sar) / np.linalg.norm(b_sar)
    spot_misfit = np.linalg.norm(G_spot.dot(m) - b_spot) / np.linalg.norm(b_spot)
    gps_misfit = np.linalg.norm(G_gps.dot(m) - b_gps) / np.linalg.norm(b_gps)
    # profiles_misfit.append(np.linalg.norm(G_profile.dot(m) - b_3) / np.linalg.norm(b_3))
    total_misfit = inv.cost / np.linalg.norm(np.concatenate((b_dens, b_spars)))
    profile_misfit = np.linalg.norm(x - y) / np.linalg.norm(y)
    gps_fs_missfit = np.linalg.norm(G_2t.dot(m) - b_2) / np.linalg.norm(b_2)
    gps_ms_missfit = np.linalg.norm(G_1t.dot(m) - b_1) / np.linalg.norm(b_1)
    return spars_misfit, dens_misfit, sar_misfit, gps_misfit, profile_misfit, gps_fs_missfit, spot_misfit, total_misfit, gps_ms_missfit




def build_CA_dens_ker1(inv, ker_array, imgary2sar = 1.0, smooth=None):
    G_spot1_ew = ker_array[2]
    G_spot1_ns = ker_array[3]

    G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0)
    G_spot1 = np.concatenate((G_spot1, np.zeros_like(G_spot1)), axis=1) / imgary2sar

    G_sar1 = ker_array[0]
    G_sar2 = ker_array[1]
    G_sar = np.concatenate((G_sar1, G_sar2), axis=0)
    G_sar = np.concatenate((G_sar, G_sar), axis=1)

    G_dens = np.concatenate((G_sar, G_spot1), axis=0)
    if smooth is not None:
        smooth1 = np.concatenate((smooth, np.zeros_like(smooth)), axis=1)
        smooth2 = np.concatenate((np.zeros_like(smooth), smooth), axis=1)
        smooth = np.concatenate((smooth1, smooth2))
        smooth1 = np.concatenate((smooth, np.zeros_like(smooth)), axis=1)
        smooth2 = np.concatenate((np.zeros_like(smooth), smooth), axis=1)
        smooth = np.concatenate((smooth1, smooth2))
        G_dens = np.concatenate((G_dens, smooth))
    return G_dens

def build_CA_dens_ker3(inv, ker_array, imgary2sar=1.0, smooth=None):
    G_spot1_ew = ker_array[0]
    G_spot1_ns = ker_array[1]

    G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0)

    G_dens = G_spot1
    if smooth is not None:
        smooth1 = np.concatenate((smooth, np.zeros_like(smooth)), axis=1)
        smooth2 = np.concatenate((np.zeros_like(smooth), smooth), axis=1)
        smooth = np.concatenate((smooth1, smooth2))
        smooth1 = np.concatenate((smooth, np.zeros_like(smooth)), axis=1)
        smooth2 = np.concatenate((np.zeros_like(smooth), smooth), axis=1)
        smooth = np.concatenate((smooth1, smooth2))
        G_dens = np.concatenate((G_dens, smooth))
    return G_dens

def build_CA_dens_ker(inv, ker_array, imgary2sar=1.0, smooth=None):
    G_spot1_ew = ker_array[2]
    G_spot1_ns = ker_array[3]

    G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0) / imgary2sar

    G_sar1 = ker_array[0]
    G_sar2 = ker_array[1]
    G_sar = np.concatenate((G_sar1, G_sar2), axis=0)

    G_dens = np.concatenate((G_sar, G_spot1), axis=0)
    if smooth is not None:
        G_dens = np.concatenate((G_dens, smooth))
    return G_dens




def solve_CA_cs_no_constraints(inv, G_kw):
    beta = G_kw['beta']
    alpha = G_kw['alpha']
    im2sar = G_kw['im2sar']


    mask1 = np.concatenate((inv.gps[0].data.E, inv.gps[0].data.N, inv.gps[0].data.Up)) != 0
    b_1 = inv.gps[0].get_data()
    w_1 = inv.gps[0].cala_whigts(mask1)
    G_1 = np.concatenate((inv.gps[0].G_ss, inv.gps[0].G_ds), axis=1)
    G_1 = G_1[mask1]
    b_1 = b_1[mask1]
    G_1w = G_1 * w_1.reshape(-1, 1)
    b_1w = b_1 * w_1 * alpha
    G_1w = alpha * np.concatenate((G_1w, np.zeros((G_1w.shape[0], G_1.shape[1]))), axis=1)


    mask2 = np.concatenate((inv.gps[1].data.E, inv.gps[1].data.N, inv.gps[1].data.Up)) != 0
    b_2 = inv.gps[1].get_data()
    w_2 = inv.gps[1].cala_whigts(mask2)
    G_2 = np.concatenate((inv.gps[1].G_ss, inv.gps[1].G_ds), axis=1)
    G_2 = G_2[mask2]
    b_2 = b_2[mask2]
    G_2w = G_2 * w_2.reshape(-1, 1)
    b_2w = b_2 * w_2 * alpha
    G_2w = alpha * np.concatenate((np.zeros((G_2w.shape[0], G_1.shape[1])), G_2w), axis=1)



    imgary2sar = im2sar
    b_spot1_ew, G_spot1_ew = inv.images[2].get_image_kersNdata()
    b_spot1_ns, G_spot1_ns = inv.images[3].get_image_kersNdata()

    G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0)
    b_spot1 = np.concatenate((b_spot1_ew, b_spot1_ns)) / imgary2sar
    G_spot1 = np.concatenate((G_spot1, np.zeros_like(G_spot1)), axis=1) / imgary2sar



    b_sar1, G_sar1 = inv.images[0].get_image_kersNdata()
    b_sar2, G_sar2 = inv.images[1].get_image_kersNdata()
    G_sar = np.concatenate((G_sar1, G_sar2), axis=0)
    G_sar = np.concatenate((G_sar, G_sar), axis=1)
    b_sar = np.concatenate((b_sar1, b_sar2))

    G = np.concatenate((G_1w, G_2w, G_sar, G_spot1), axis=0)
    bw = np.concatenate((b_1w, b_2w, b_sar, b_spot1))
    return bw, G

def solve_CA_cs(inv, G_kw):
    beta = G_kw['beta']
    alpha = G_kw['alpha']
    im2sar = G_kw['im2sar']


    mask1 = np.concatenate((inv.gps[0].data.E, inv.gps[0].data.N, inv.gps[0].data.Up)) != 0
    b_1 = inv.gps[0].get_data()
    w_1 = inv.gps[0].cala_whigts(mask1)
    G_1 = np.concatenate((inv.gps[0].G_ss, inv.gps[0].G_ds), axis=1)
    G_1 = G_1[mask1]
    b_1 = b_1[mask1]
    G_1w = G_1 * w_1.reshape(-1, 1)
    b_1w = b_1 * w_1 * alpha
    G_1w = alpha * np.concatenate((G_1w, np.zeros((G_1w.shape[0], G_1.shape[1]))), axis=1)


    mask2 = np.concatenate((inv.gps[1].data.E, inv.gps[1].data.N, inv.gps[1].data.Up)) != 0
    b_2 = inv.gps[1].get_data()
    w_2 = inv.gps[1].cala_whigts(mask2)
    G_2 = np.concatenate((inv.gps[1].G_ss, inv.gps[1].G_ds), axis=1)
    G_2 = G_2[mask2]
    b_2 = b_2[mask2]
    G_2w = G_2 * w_2.reshape(-1, 1)
    b_2w = b_2 * w_2 * alpha
    G_2w = alpha * np.concatenate((np.zeros((G_2w.shape[0], G_1.shape[1])), G_2w), axis=1)



    imgary2sar = im2sar
    b_spot1_ew, G_spot1_ew = inv.images[2].get_image_kersNdata()
    b_spot1_ns, G_spot1_ns = inv.images[3].get_image_kersNdata()

    G_spot1 = np.concatenate((G_spot1_ew, G_spot1_ns), axis=0)
    b_spot1 = np.concatenate((b_spot1_ew, b_spot1_ns)) / imgary2sar
    G_spot1 = np.concatenate((G_spot1, np.zeros_like(G_spot1)), axis=1) / imgary2sar



    b_sar1, G_sar1 = inv.images[0].get_image_kersNdata()
    b_sar2, G_sar2 = inv.images[1].get_image_kersNdata()
    G_sar = np.concatenate((G_sar1, G_sar2), axis=0)
    G_sar = np.concatenate((G_sar, G_sar), axis=1)
    b_sar = np.concatenate((b_sar1, b_sar2))

    G = np.concatenate((G_1w, G_2w, G_sar, G_spot1), axis=0)

    ks = np.array([len(p.sources) for p in inv.plains])




    ### constraint pplains 4,6,13 to 0 for the fs
    zero_constrain = np.zeros((ks[4] + ks[6] + ks[13], G.shape[1]))
    b_zero_constrain = np.zeros(ks[4] + ks[6] + ks[13])
    for i in range(ks[4]):
        zero_constrain[i, np.sum(ks) + np.sum(ks[:4]) + i] = 1
    for i in range(ks[6]):
        zero_constrain[ks[4] + i, np.sum(ks) + np.sum(ks[:6]) + i] = 1
    for j in range(ks[13]):
        zero_constrain[ks[4] + ks[6]+j, np.sum(ks) + np.sum(ks[:13]) + j] = 1
    zero_constrain *= 1e6



    # constraint plains 1-18 to 0 for the fs
    # ppp = 18
    # identity = np.identity(np.sum(ks[:ppp]))
    # zero_constrain = np.concatenate((np.zeros((np.sum(ks[:ppp]), np.sum(ks))), identity, np.zeros((identity.shape[0], np.sum(ks[ppp:])))), axis=1)
    # b_zero_constrain = np.zeros(np.sum(ks[:ppp]))
    # zero_constrain *= 1e6

    # # constraint segment 2 to 0 for the fs
    # zero_constrain = np.zeros((6, np.sum(ks) * 2))
    # zero_constrain[0, np.sum(ks) + 4] = 1.0
    # zero_constrain[1, np.sum(ks) + 5] = 1.0
    # zero_constrain[2, np.sum(ks) + 6] = 1.0
    # zero_constrain[3, np.sum(ks) + 14] = 1.0
    # zero_constrain[4, np.sum(ks) + 15] = 1.0
    # zero_constrain[5, np.sum(ks) + 16] = 1.0
    # b_zero_constrain = np.zeros(6)
    # zero_constrain *= 1e6





    def constain_surface(plains, i_l, i_r):
        if not isinstance(plains, list):
            plains = [plains]
        n = np.sum([len(c.sources) for c in plains])
        constraint = np.zeros((0, n))
        k = 0
        for p in plains:
            for s in p.sources:
                if s.depth_t <= 0.5:
                    c = np.zeros((1, n))
                    c[0, k] = 1
                    constraint = np.concatenate((constraint, c), axis=0)
                k += 1
        constraint = np.concatenate((np.zeros((constraint.shape[0], np.sum(ks[:i_l]))), constraint,
            np.zeros((constraint.shape[0], np.sum(ks[i_r:])))), axis=1)

        return constraint
    const_s1 = constain_surface(inv.plains[19], 19, 20)
    const_s2 = constain_surface(inv.plains[20], 20, 21)
    const_s3 = constain_surface(inv.plains[21], 21, 22)
    const_nw = constain_surface(inv.plains[:19], 0, 19)

    const_s =  np.concatenate((const_s1, const_s2, const_s3), axis=0)

    disp_s = np.array([0.9226381956589431, 1.2225717588576088, 0.8985328915287425, 0.6385295812081863, 0.6215600130290337, 0.6428820928777036, 0.7375838786029231])
    sigma = np.array([0.8439504629111646, 0.7171534437334907, 0.6916827780891992, 0.6297644795186776, 0.6500183551428386, 0.7197180758218581, 0.6531201924095549])
    const_sw = (1 / (sigma * np.sum(1/sigma)))
    const_s = const_s * const_sw.reshape(-1, 1)
    disp_s *= const_sw
    const_s = np.concatenate((const_s, const_nw))
    const_s = np.concatenate((np.zeros_like(const_s), const_s), axis=1)
    disp_s = np.concatenate((disp_s, np.zeros(const_nw.shape[0])))




    const_s_ms = const_s3
    const_s_ms = np.concatenate((const_s_ms, np.zeros_like(const_s_ms)), axis=1)



    smoothing1 = beta * inv.S
    smoothing1_1 = np.concatenate((smoothing1, np.zeros(smoothing1.shape)), axis=1)
    smoothing1_2 = np.concatenate((np.zeros(smoothing1.shape), smoothing1), axis=1)
    smoothing = np.concatenate((smoothing1_1, smoothing1_2), axis=0)




    bw = np.concatenate((b_1w, b_2w, b_sar, b_spot1))

    Gs = np.concatenate((G, smoothing, zero_constrain, const_s, const_s_ms), axis=0)
    bs = np.concatenate((bw, np.zeros(smoothing.shape[0]), b_zero_constrain, disp_s, np.zeros(const_s_ms.shape[0])))
    return bs, Gs



def plot_sources_CA(inv, cmap_max=None, cmap='jet', set=None, epicenter=None):
    if set is None:
        set = inv.gps[0]
    def plot_one_plain(plain, slip, ax, cmap, norm):
        plain.plot_sources_2d(ax, slip, cmap, norm)
        ax.set_xlim(0, plain.plain_length)
        ax.set_ylim(-plain.total_width, 0)

    def plot_plain(plain, ss , ds, total, axs, cmap, norm):
        plot_one_plain(plain, ss, axs[0], cmap, norm)
        plot_one_plain(plain, ds, axs[1], cmap, norm)
        plot_one_plain(plain, total, axs[2], cmap, norm)

    ratio = np.array([p.plain_length for p in set.plains])
    ratio = ratio / ratio.max()
    if inv.solution is None:
        fig, axs = plt.subplots(1, ratio.shape[0], gridspec_kw={'width_ratios': ratio})
        for p, ax in zip(set.plains, axs.flatten()):
            plot_one_plain(p, None, ax, None, None)
    else:

        ks = np.array([len(p.sources) for p in set.plains])
        n = np.sum(ks)
        MS = inv.solution[0 : n * 2]
        FS = inv.solution[n * 2 : n * 4]


        for m, str in zip([MS, FS], ['MS', 'FS']):
            ss = m[:n]
            ds = m[n:]

            total = np.sqrt(ss**2 + ds**2)

            cmap = cm.get_cmap(cmap)
            norm = mlb.colors.Normalize(0, total.max())
            fig, axs = plt.subplots(3, ks.shape[0], gridspec_kw={'width_ratios': ratio})

            if epicenter is not None:
                from gdal_utils import shear_hat
                p = 1
                e1 = shear_hat(np.pi / 2 - np.deg2rad(set.plains[p].strike), np.deg2rad(set.plains[p].dip), 0)
                e2 = shear_hat(np.pi / 2 - np.deg2rad(set.plains[p].strike), np.deg2rad(set.plains[p].dip), -np.pi/2)
                v1 = np.array([set.plains[p].plain_cord[0], set.plains[p].plain_cord[1], set.plains[p].plain_cord[2]]).reshape(-1, 1)
                x = Inversion.dd2m(epicenter[0] - inv.images[0].lon, lat=inv.images[0].lat) * 1e-3
                y = Inversion.dd2m(epicenter[1] - inv.images[0].lat) * 1e-3
                v2 = np.array([x, y, epicenter[2]]).reshape(-1, 1)
                t1 = np.sum(e1*(v2-v1))
                t2 = np.sum(e2*(v2-v1))
                for ax in axs[:, p]:
                    ax.scatter(t1, t2, marker='*', color='k', s=20, zorder=2)

            for i in range(ks.shape[0]):
                ss_p = ss[np.sum(ks[0:i]): np.sum(ks[0:i])+ks[i]]
                ds_p = ds[np.sum(ks[0:i]): np.sum(ks[0:i])+ks[i]]
                tot_p = total[np.sum(ks[0:i]): np.sum(ks[0:i])+ks[i]]
                plot_plain(set.plains[i], ss_p, ds_p, tot_p, axs[:, i], cmap, norm)



            axs[0, 0].set_ylabel('Strike slip')
            axs[1, 0].set_ylabel('dip slip')
            axs[2, 0].set_ylabel('Total slip')

            cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
            cmmapable = cm.ScalarMappable(norm, cmap)
            cmmapable.set_array(np.linspace(0, total.max()))
            cbar = plt.colorbar(cmmapable, orientation='horizontal', cax=cax)
            cbar.set_label('slip [m]')

            fig.suptitle(str)



def plot_sources_CA_2d(inv,  set, cmap_max=None, cmap='jet', epicenter=None, slip=None, cbar=True, fig=None):
    def plot_one_plain(plain, slip, ax, cmap, norm, title):
        plain.plot_sources_2d(ax, slip, cmap, norm)
        ax.set_xlim(0, plain.plain_length)
        ax.set_ylim(-plain.total_width, 0)
        ax.set_title(title)

    ratio = np.array([p.plain_length for p in set.plains])
    ratio = ratio / ratio.max()
    if inv.solution is None and slip is None:
        fig, axs = plt.subplots(1, ratio.shape[0], gridspec_kw={'width_ratios': ratio})
        for p, ax in zip(set.plains, axs.flatten()):
            plot_one_plain(p, None, ax, None, None, '')
    else:
        if slip is None:
            slip = inv.solution
        ks = np.array([len(p.sources) for p in set.plains])


        cmap = cm.get_cmap(cmap)
        if cmap_max is None:
            cmap_max = slip.max()
        norm = mlb.colors.Normalize(0, cmap_max)
        if fig is None:
            fig = plt.figure()
        axs = fig.subplots(1, ks.shape[0], gridspec_kw={'width_ratios': ratio})

        if epicenter is not None:
            from gdal_utils import shear_hat
            p=1
            e1 = shear_hat(np.pi / 2 - np.deg2rad(set.plains[p].strike), np.deg2rad(set.plains[p].dip), 0)
            e2 = shear_hat(np.pi / 2 - np.deg2rad(set.plains[p].strike), np.deg2rad(set.plains[p].dip), -np.pi/2)
            v1 = np.array([set.plains[p].plain_cord[0], set.plains[p].plain_cord[1], set.plains[p].plain_cord[2]]).reshape(-1, 1)
            x = Inversion.dd2m(epicenter[0] - inv.images[0].lon, lat=inv.images[0].lat) * 1e-3
            y = Inversion.dd2m(epicenter[1] - inv.images[0].lat) * 1e-3
            v2 = np.array([x, y, epicenter[2]]).reshape(-1, 1)
            t1 = np.sum(e1*(v2-v1))
            t2 = np.sum(e2*(v2-v1))
            axs[p].scatter(t1, t2, marker='*', color='k', s=20, zorder=2)

        for i in range(ks.shape[0]):
            ss_p = slip[np.sum(ks[0:i]): np.sum(ks[0:i])+ks[i]]
            plot_one_plain(set.plains[i], ss_p, axs[i], cmap, norm, i+1)

        for i in range(1, ks.shape[0]):
            axs[i].set_yticks([])


        if cbar:
            cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
            cmmapable = cm.ScalarMappable(norm, cmap)
            cmmapable.set_array(np.linspace(0, slip.max()))
            cbar = plt.colorbar(cmmapable, orientation='horizontal', cax=cax)
            cbar.set_label('slip [m]')


def plot_sources_CA_cs(inv, cmap_max=None, cmap='jet', set=None, epicenter=None, slip=None):
    if set is None:
        set = inv.gps[0]

    if inv.solution is None and slip is None:
        inv.plot_sources_CA_2d(set)
    else:
        if slip is None:
            slip = inv.solution
        ks = np.array([len(p.sources) for p in set.plains])
        n = np.sum(ks)
        MS = slip[0: n]
        FS = slip[n: n * 2]


        for m, str in zip([MS, FS], ['MS', 'FS']):
            fig  = plt.figure()
            inv.plot_sources_CA_2d(set, cmap_max, cmap, epicenter, slip=m, cbar=True, fig=fig)
            fig.suptitle(str)








def CA_moment(inv):
    n = np.sum([len(p.sources) for p in inv.gps[0].plains])
    MS = inv.solution[0: n]
    FS = inv.solution[n: n * 2]
    ms_moment = 0
    fs_moment = 0
    shift = 0
    for p in inv.gps[0].plains:
        k = len(p.sources)
        ms_moment += p.seismic_moment(MS[shift:k + shift], np.zeros(k))
        fs_moment += p.seismic_moment(FS[shift:k + shift], np.zeros(k))
        shift += k
    return (np.log10(ms_moment) - 9.05) / 1.5, (np.log10(fs_moment) - 9.05) / 1.5

def roughness_missfit(inv, in_alpha, min, max, num=30, return_all=False, im2sar=10.0):
    betas = np.logspace(min, max, num)
    spars_misfit = []
    dens_misfit = []
    sar_misfit = []
    spot_misfit = []
    gps_misfit = []
    profiles_misfit = []
    cost = []
    total_misfit =[]
    moment_fs = []
    moment_ms = []
    roughness = []
    for beta in betas:
        print(beta)
        inv.solve_CA_cs(beta, in_alpha, im2sar)
        cost.append(inv.cost)


        mask1 = np.concatenate((inv.gps[0].data.E, inv.gps[0].data.N, inv.gps[0].data.Up)) != 0
        b_1 = inv.gps[0].get_data()
        G_1 = np.concatenate((inv.gps[0].G_ss, inv.gps[0].G_ds), axis=1)
        G_1 = G_1[mask1]
        b_1 = b_1[mask1]
        w_1 = inv.gps[0].cala_whigts(mask1)

        mask2 = np.concatenate((inv.gps[1].data.E, inv.gps[1].data.N, inv.gps[1].data.Up)) != 0
        b_2 = inv.gps[1].get_data()
        G_2 = np.concatenate((inv.gps[1].G_ss, inv.gps[1].G_ds), axis=1)
        G_2 = G_2[mask2]
        b_2 = b_2[mask2]
        w_2 = inv.gps[1].cala_whigts(mask2)




        G_1t = np.concatenate((G_1, np.zeros((G_1.shape[0], G_2.shape[1]))), axis=1)
        G_2t = np.concatenate((np.zeros((G_2.shape[0], G_1.shape[1])), G_2), axis=1)


        G_gps = np.concatenate((G_1t, G_2t)) * np.concatenate((w_1, w_2)).reshape(-1, 1)
        b_gps = np.concatenate((b_1, b_2)) * np.concatenate((w_1, w_2))


        G_spars = G_gps
        b_spars = b_gps

        b_spot, G_spot = inv.get_sar_inv_pars(False, (2, 4))
        b_spot = b_spot / im2sar
        G_spot = np.concatenate((G_spot, np.zeros_like(G_spot)), axis=1) / im2sar


        b_sar, G_sar = inv.get_sar_inv_pars(False, (0, 2))
        G_sar = np.concatenate((G_sar, G_sar), axis=1)


        G_dens = np.concatenate((G_sar, G_spot), axis=0)
        b_dens = np.concatenate((b_sar, b_spot))

        m = inv.solution
        ks = np.array([len(p.sources) for p in inv.gps[0].plains])
        k_1 = np.sum(ks[:-1])
        k_2 = ks[-1]
        mask_rl = np.concatenate((np.ones(k_1, dtype=np.bool), np.zeros(k_2, dtype=np.bool),
                                  np.ones(k_1, dtype=np.bool), np.zeros(k_2, dtype=np.bool),
                                  np.ones(k_1, dtype=np.bool), np.zeros(k_2, dtype=np.bool),
                                  np.ones(k_1, dtype=np.bool), np.zeros(k_2, dtype=np.bool)))

        spars_misfit.append(np.linalg.norm(G_spars.dot(m) - b_spars) / np.linalg.norm(b_spars))
        # m[mask_rl] *= -1
        dens_misfit.append(np.linalg.norm(G_dens.dot(m) - b_dens) / np.linalg.norm(b_dens))
        sar_misfit.append(np.linalg.norm(G_sar.dot(m) - b_sar) / np.linalg.norm(b_sar))
        spot_misfit.append(np.linalg.norm(G_spot.dot(m) - b_spot) / np.linalg.norm(b_spot))
        gps_misfit.append(np.linalg.norm(G_gps.dot(m) - b_gps) / np.linalg.norm(b_gps))
        # profiles_misfit.append(np.linalg.norm(G_profile.dot(m) - b_3) / np.linalg.norm(b_3))
        total_misfit.append(inv.cost / np.linalg.norm(np.concatenate((b_dens, b_spars))))


        ms_mw, fs_mw = inv.CA_moment_cs()
        moment_fs.append(fs_mw)
        moment_ms.append(ms_mw)

        smoothing1 = inv.gps[0].get_smoothing_no_bounds(inv.zero_pad)
        smoothing1_1 = np.concatenate((smoothing1, np.zeros(smoothing1.shape)), axis=1)
        smoothing1_2 = np.concatenate((np.zeros(smoothing1.shape), smoothing1), axis=1)
        smoothing = np.concatenate((smoothing1_1, smoothing1_2), axis=0)

        roughness.append(np.sum(np.abs(smoothing.dot(inv.solution))) / inv.solution.shape[0])


    if return_all:
        return spars_misfit, dens_misfit, sar_misfit, spot_misfit, gps_misfit, profiles_misfit, total_misfit, betas, moment_fs, moment_ms, roughness
    return spars_misfit, dens_misfit