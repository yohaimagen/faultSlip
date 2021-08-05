import os, argparse, re

arg_parser = argparse.ArgumentParser(description="This program generate an inversion model parameter file interactively or from a text file")
arg_parser.add_argument("path", help="path to save the output files with no .json extension")
arg_parser.add_argument("--input", help="path to input file for the script")
args = arg_parser.parse_args()

param_in = ''
if args.input:
    inps = open(args.input)
def col_input(str):
    global param_in
    if args.input:
        inp = inps.readline().strip()
        print(str + inp)
        return inp
    else:
        inp = input(str)
        param_in += '%s\n' %inp
        return inp


par_file = \
'''{
"global_parameters":
\t{
\t"poisson_ratio":%s,
\t"shear_modulus":%s,
\t"smooth":%s,
\t"dip_element":%s,
\t"strike_element":%s,
\t"compute_mean":%s,
\t"boundary_constrained":%s
\t},\n'''

poisson_ratio = col_input("poisson_ratio:")
shear_modulus = col_input("shear_modulus:")
smooth = col_input("smooth:")
dip_element = col_input("dip_element:")
strike_element = col_input("strike_element:")
compute_mean = col_input("compute_mean:")
boundary_constrained = col_input("boundary_constrained:")
par_file = par_file %(poisson_ratio, shear_modulus, smooth, dip_element, strike_element, compute_mean, boundary_constrained)

more_plains = True
plain_str = \
'''
\t\t\t"plain%d":
\t\t\t\t{
\t\t\t\t"dip":%s,
\t\t\t\t"strike":%s,
\t\t\t\t"plain_cord":[%s],
\t\t\t\t"plain_length":%s,%s
\t\t\t\t"width":[%s],
\t\t\t\t"num_sub_stk":[%s],
\t\t\t\t"smooth_up":%s,
\t\t\t\t"strike_element":%s,
\t\t\t\t"dip_element":%s,
\t\t\t\t "total_width":%s
\t\t\t\t}'''
plains = []
plain_num = 1
while more_plains:
    dip = col_input("dip:")
    strike = col_input("strike")
    plain_cord = col_input("plain_cord:")
    plain_length = col_input("plain_length:")
    source_file = col_input("source_file:")
    if source_file == 'n':
        width = col_input('width:')
        num_sub_stk = col_input("num_sub_stk:")
        source_file = ''
    else:
        source_file = '\n\t\t\t\t"sources_file":"%s",' %source_file
        width = col_input('width:')
        num_sub_stk = col_input("num_sub_stk:")
    # smooth_right = col_input("smooth_right:")
    # smooth_left = col_input("smooth_left:")
    smooth_up = col_input("smooth_up:")
    strike_element = col_input("strike_element:")
    dip_element = col_input("dip_element:")
    total_width = col_input("total_width:")
    plain_t = plain_str %(plain_num, dip, strike, plain_cord, plain_length, source_file, width, num_sub_stk, smooth_up, strike_element, dip_element, total_width)
    plains.append(plain_t)
    another_plain = col_input("do you want anther plain?")
    if another_plain == 'n' or another_plain == '':
        more_plains = False
    plain_num += 1
plains_str = ','.join(plains)

in_images = col_input("do you want images in your model?")
if in_images == 'y':

    image_str = \
'''
\t"image%d":
\t\t{
\t\t"disp_file":"/%s",
\t\t"lon":%s,
\t\t"lat":%s,
\t\t"x_pixel":%s,
\t\t"y_pixel":%s,
\t\t"incidence_angle":%s,
\t\t"azimuth":%s,
\t\t"station":"%s",
\t\t"station_num":%s,
\t\t"origin_y":%s,
\t\t"origin_x":%s,
\t\t"plains":
\t\t\t{%s
\t\t\t}
\t\t}'''

    images = []
    image_num = 1
    more_images = True
    while more_images:
        disp_file = col_input("disp_file:")
        lon = col_input("lon:")
        lat = col_input("lat:")
        x_pixel = col_input("x_pixel:")
        y_pixel = col_input("y_pixel:")
        incidence_angle = col_input("incidence_angle:")
        azimuth = col_input("azimuth:")
        station = col_input("station:")
        if station == 'uniform':
            station_num = col_input("station_size:")
        else:
            station_num = col_input("station_size:")
        origin_y = col_input("origin_y:")
        origin_x = col_input("origin_x:")
        image_t = image_str %(image_num, disp_file, lon, lat, x_pixel, y_pixel, incidence_angle, azimuth, station, station_num, origin_y, origin_x, plains_str)
        images.append(image_t)
        image_num += 1
        another_image = col_input("do you want another image?")
        if another_image == 'n':
            more_images = False

    images_str = ','.join(images)
    images_str ='"images":\n\t{%s\n\t}' %images_str
else:
    images_str = ''

in_gps =col_input("do you want gps data in your model?")

if in_gps == 'y':
    gps_str = \
'''
\t"gps_set_%d":
\t\t{
\t\t"data":"%s",
\t\t"plains":
\t\t\t{%s
\t\t\t}
\t\t}'''

    gpses =[]
    gps_num = 1
    more_gps = True
    while more_gps:
        data = col_input("data:")
        gps_t = gps_str %(gps_num, data, plains_str)
        gpses.append(gps_t)
        another_gps = col_input("do you want another gps set?")
        gps_num += 1
        if another_gps == 'n':
            more_gps = False

    gpses_str = ','.join(gpses)
    gpses_str = '"gps":\n\t{%s\n\t}' %gpses_str
else:
    gpses_str = ''

if images_str != '' and gpses_str != '':
    par_file += images_str + ',\n' + gpses_str +'\n}'
elif images_str != '' and gpses_str == '':
    par_file += images_str + '\n}'
elif images_str == '' and gpses_str != '':
    par_file += gpses_str + '\n}'

with open(args.path + '.json', 'w') as f:
    f.write(par_file)

if not args.input:
    with open(args.path + '_in', 'w') as f:
        f.write(param_in)