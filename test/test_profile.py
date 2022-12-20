import numpy as np
import sys
sys.path.insert(0,'/Users/yohai/workspace/faultSlip/')
from faultSlip.profiles import Profile
from faultSlip.inversion import Inversion
import matplotlib.pyplot as plt


par = '''{
"global_parameters":
	{
	"poisson_ratio":0.25,
	"shear_modulus":30e9,
	"smooth":0.1,
	"dip_element":0,
	"strike_element":1,
	"open_element":0,
	"compute_mean":false,
	"boundary_constrained":false,
	"origin_lon":-118.1,
	"origin_lat":35.3
	},
"plains":
	{
	"plain1":
		{
		"dip":90.0,
		"strike":137.0,
		"plain_cord":[30.26, 70.02, 0.1],
		"plain_length":60.0,
		"width":[1.3],
		"num_sub_stk":[6],
		"smooth_up":[],
		"strike_element":-1,
		"dip_element":0,
			"total_width":10
		},
    "plain2":
		{
		"dip":90.0,
		"strike":137.0,
		"plain_cord":[30.26, 70.02, 1.4],
		"plain_length":60.0,
		"width":[5.3],
		"num_sub_stk":[4],
		"smooth_up":[],
		"strike_element":-1,
		"dip_element":0,
			"total_width":10
		},
    "plain3":
		{
		"dip":90.0,
		"strike":137.0,
		"plain_cord":[30.26, 70.02, 6.7],
		"plain_length":60.0,
		"width":[5.3],
		"num_sub_stk":[4],
		"smooth_up":[],
		"strike_element":-1,
		"dip_element":0,
			"total_width":10
		},
	"plain4":
		{
		"dip":90.0,
		"strike":137.0,
		"plain_cord":[30.26, 70.02, 12.0],
		"plain_length":60.0,
		"width":[9.2],
		"num_sub_stk":[4],
		"smooth_up":[],
		"strike_element":-1,
		"dip_element":0,
			"total_width":10
		}
	},
"profiles":
    {
    "profile1":
        {
            "x":"/Users/yohai/workspace/east_california/okada/resamp_cx_2.npy",
            "y":"/Users/yohai/workspace/east_california/okada/resamp_cy_2.npy",
            "data":"/Users/yohai/workspace/east_california/okada/resamp_disp2.npy",
            "heading":-13.13,
            "incidence_angle":0
        },
    "profile2":
        {
            "x":"/Users/yohai/workspace/east_california/okada/resamp_cx_3.npy",
            "y":"/Users/yohai/workspace/east_california/okada/resamp_cy_3.npy",
            "data":"/Users/yohai/workspace/east_california/okada/resamp_disp3.npy",
            "heading":-13.13,
            "incidence_angle":0
        }
    },
"gps":
    {
    "gps1":
        {
            "data":"/Users/yohai/workspace/east_california/static_inversion/gnss_disp.csv",
            "origin_lat":35.3,
            "origin_lon":-118.1
        }
    }
}'''

with open('__t.par', 'w') as f:
    f.write(par)
inv = Inversion('./__t.par')


inv.build_kers()
# inv.plot_sources()
def get_g(inv, G_kw):
    alpha = G_kw['alpha']
    Gp1 = np.concatenate((inv.profiles[0].G_ss, inv.profiles[0].G_ds, inv.profiles[0].G_o), axis=1)
    Gp2 = np.concatenate((inv.profiles[1].G_ss, inv.profiles[1].G_ds, inv.profiles[1].G_o), axis=1)
    Ggnss = inv.gps[0].G_ss[:inv.gps[0].data.shape[0] * 2]
    G = np.concatenate((Gp1, Gp2, Ggnss * alpha), axis=0)
    bgnss = np.concatenate((inv.gps[0].data.E.values, inv.gps[0].data.N.values))
    b = np.concatenate((inv.profiles[0].data, inv.profiles[1].data, alpha * bgnss))
    return b, G
def calc_misfit(inv, slip):
    Gp1 = np.concatenate((inv.profiles[0].G_ss, inv.profiles[0].G_ds, inv.profiles[0].G_o), axis=1)
    Gp2 = np.concatenate((inv.profiles[1].G_ss, inv.profiles[1].G_ds, inv.profiles[1].G_o), axis=1)
    Ggnss = inv.gps[0].G_ss[:inv.gps[0].data.shape[0] * 2]
    G = np.concatenate((Gp1, Gp2, Ggnss), axis=0)
    bgnss = np.concatenate((inv.gps[0].data.E.values, inv.gps[0].data.N.values))
    b = np.concatenate((inv.profiles[0].data, inv.profiles[1].data, bgnss))
    return np.linalg.norm(G.dot(slip) - b, 2) / b.shape[0]


# alphas = np.logspace(-3, 1, 1000)
# costs = []
# for alpha in alphas:    
#     inv.solve_g(get_g, {'alpha':alpha})
#     costs.append(calc_misfit(inv, inv.solution))
# plt.figure()
# plt.plot(alphas, costs)
# plt.xscale('log')
# print(alphas[np.argmin(costs)])

inv.solve_g(get_g, {'alpha':1})
inv.plot_profiles(slip=inv.solution)
inv.gps[0].plot_en(slip=inv.solution)
inv.plot_sources(slip=np.concatenate((inv.solution, np.zeros_like(inv.solution))))
plt.show()
