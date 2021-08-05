from faultSlip.inversion import Inversion
import numpy as np

inv_parms = '''{
"global_parameters":
	{
	"poisson_ratio":0.25,
	"shear_modulus":30e9,
	"smooth":0.1,
	"dip_element":1,
	"strike_element":1,
    "open_element":0,
	"compute_mean":false,
	"boundary_constrained":false
	},
"plains":
	{
	"plain1":
		{
		"dip":%f,
		"strike":%f,
		"plain_cord":[%f, %f, %f],
		"plain_length":%f,
		"width":[%f],
		"num_sub_stk":[1],
		"smooth_up":[],
		"strike_element":1,
		"dip_element":1,
		 "total_width":10
		}
	},
"gps":
	{
	"gps_set_1":
		{
		"data":"/Users/yohai/workspace/GPS_processing/ibaraki/off_seis_syn/gps.csv"
		}
	}
}
'''

inv_parms2 = '''{
"global_parameters":
	{
	"poisson_ratio":0.25,
	"shear_modulus":30e9,
	"smooth":0.1,
	"dip_element":1,
	"strike_element":1,
    "open_element":0,
	"compute_mean":false,
	"boundary_constrained":false
	},
"plains":
	{
	"plain1":
		{
		"dip":%f,
		"strike":%f,
		"plain_cord":[%f, %f, %f],
		"plain_length":%f,
		"width":[%f],
		"num_sub_stk":[1],
		"smooth_up":[],
		"strike_element":1,
		"dip_element":-1,
		 "total_width":10
		}
	},
"gps":
	{
	"gps_set_1":
		{
		"data":"/Users/yohai/workspace/GPS_processing/ibaraki/off_seis_syn/gps.csv"
		}
	}
}
'''

params = {'dip': 29.462080004723077,
 'strike': 33.17450557382973,
 'x': 153.08304281277867,
 'y': 79.0166098922095,
 'depth': 14.304054329855298,
 'width': 69.57031053995074,
 'length': 59.18350980958724}

def get_inv_1d(par, dip, strike, x, y, depth, width, length):
    pars = par%(dip, strike, x, y, depth, length, width)
    with open(f'./__params.json', 'w') as f:
        f.write(pars)
    return Inversion(f'./__params.json')

inv1 = get_inv_1d(inv_parms, **params)
inv1.build_kers()

inv2 = get_inv_1d(inv_parms2, **params)
inv2.build_kers()

print('')


