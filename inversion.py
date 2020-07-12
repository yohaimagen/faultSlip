import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mlb
import matplotlib.cm as cm
from scipy import optimize
import numpy as np
import json
from ps import Point_sources
from image import Image
from gps_set import Gps
from seismicity import Seismisity
from plain import Plain
from dists.dist import neighbors




class Inversion:
    """
    Inversion class  is the base class for inverting geodetic data to slip along faults

    Args:
        par_file(string): path to inversion parameter file

    Attributes:
        poisson_ratio(float): poisson ratio for the elastic half space medium
        shear_modulus(float): shear_modulus for the elastic half medium
        solution(ndarray): slip solution of the model with shape (n,) where  n is the number of dislocations in the model
        images(list): list of dens datasets
        gps(list): list of gps datasets
    """
    def __init__(self, par_file):
        with open(par_file, 'r') as f:
            in_data = json.load(f)
        global_parameters = in_data['global_parameters']
        self.poisson_ratio = global_parameters['poisson_ratio']
        self.shear_modulus = global_parameters['shear_modulus']
        self.dip_element = global_parameters['dip_element']
        self.strike_element = global_parameters['strike_element']
        self.compute_mean = global_parameters['compute_mean']
        self.solution = None
        self.cost = None
        self.images = []
        self.plains = []
        self.sources_mat = None
        if "plains" in in_data.keys():
            def plains_sort(p1):
                return float(p1.split('n')[1])

            for key in sorted(in_data['plains'], key=plains_sort):
                self.plains.append(Plain(**in_data['plains'][key]))
        if 'images' in in_data.keys():
            def image_num(image):
                return float(image.split('e')[1])
            for key in sorted(in_data['images'], key=image_num):
                self.images.append(Image(plains=self.plains, **in_data['images'][key]))
        self.gps = []
        if 'gps' in in_data.keys():
            for key in sorted(in_data['gps']):
                self.gps.append(Gps(**in_data['gps'][key]))
        self.point_sources = []
        if 'point_sources' in in_data.keys():

            self.point_sources = Point_sources(in_data['point_sources'])
        self.seismisity = []
        if 'seismicity' in in_data.keys():
            def seismisty_num(seismisty):
                return float(seismisty.split('y')[1])
            for key in sorted(in_data['seismicity'], key=seismisty_num):
                self.seismisity.append(Seismisity(**in_data['seismicity'][key]))
        if type(in_data['global_parameters']['smooth']) is str:
            print('load_smoothing from %s' %in_data['global_parameters']['smooth'])
            self.S = np.load(in_data['global_parameters']['smooth'])
        else:
            self.S = self.new_smoothing()


    @staticmethod
    def m2dd(m, lat=0):
        """
        transform distance in meter to degrees

        Args:
            m: distance in meters
            lat: for converting longitude distances insert the latitude, default 0

        Returns:
            distance in degrees

        """
        return m / (111319.9 * np.cos(np.deg2rad(lat)))
    @staticmethod
    def dd2m(dd, lat=0):
        """
        transform distance in degrees to meters

        Args:
            dd(float): distance in degrees
            lat(float): for converting longitude distances insert the latitude, default 0
        Returns:
            dist: distance in meters
        """
        return dd * (111319.9 * np.cos(np.deg2rad(lat)))

    def build_kers(self):
        """building all subsets dataset elastic kernels"""
        for img in self.images:
            img.build_kernal(self.strike_element, self.dip_element, self.plains)
        for gps in self.gps:
            gps.build_ker(self.strike_element, self.dip_element, self.plains)
        for seismisity in self.seismisity:
            seismisity.build_ker(self.strike_element, self.dip_element, self.plains)

    def calculate_station_disp(self):
        """calculate the displacement for each data point"""
        if self.compute_mean:
            for img in self.images:
                img.calculate_station_disp_weight()
        else:
            for img in self.images:
                img.calculate_station_disp()

    def drop_nan_station(self):
        """drop all nan value data point"""
        for img in self.images:
            img.drop_nan_station()

    def drop_zero_station(self):
        """drop all zero value data pints"""
        for img in self.images:
            img.drop_zero_station()

    def drop_crossing_station(self):
        """droop all data points which cross the fault trace"""
        for img in self.images:
            img.drop_crossing_station(self.plains)

    def clean_solve(self):
        """solve the inversion just for dens data set without any constraint"""
        img_kers = []
        b = []
        for i, img in enumerate(self.images):
            G_img = img.get_ker(zero_pad=0, compute_mean=self.compute_mean)
            img_kers.append(G_img)
            b.append(img.get_disp(self.compute_mean))
        b = np.concatenate(b)
        ker = np.concatenate(img_kers)
        sol = optimize.nnls(ker, b)
        self.solution = sol[0]
        self.cost = sol[1]





    def get_sar_inv_pars(self, include_offset=True, subset=None):
        """
        build the elastic kernel and displacement for a subset of the dens datasets images

        Args:
            include_offset(bool): if True contain a line which solve for a constant offset of all the image
            subset: an iterable object contains the indexes of the datasets subset, defualt all dens datasets

        Returns:
            b_W: whited data
            G_w whited elastic kernel

        """
        if subset is None:
            subset = range(len(self.images))
        img_kers = []
        img_offset = []
        b = []
        w = []
        for i in subset:
            G_img = self.images[i].get_ker(zero_pad=0, compute_mean=False)
            img_kers.append(G_img)
            img_offset.append(np.concatenate([np.zeros((G_img.shape[0], i)), np.ones((G_img.shape[0], 1)),
                                              np.zeros((G_img.shape[0], len(self.images) - 1 - i))], axis=1))
            b.append(self.images[i].get_disp(False))
            w.append(self.images[i].stations_mat[:, 5])
        b = np.concatenate(b)
        ker = np.concatenate(img_kers)
        offset = np.concatenate(img_offset)
        w = np.concatenate(w)
        G_w = ker * w.reshape(-1, 1)
        b_w = b * w
        if include_offset:
            G_w = np.concatenate((G_w, offset), axis=1)
        return b_w, G_w

    def solve(self, beta=None, smoothing_mat=None):
        """
        solve inversion for dens data set alone with smoothing for min[Am - b] m >= 0

        Args:
            beta(float): smoothing coefficient
            smoothing_mat(ndarray): smoothing operator, if None that first order spatial derivative operator is considered

        """
        img_kers = []
        img_offset = []
        b = []
        for i, img in enumerate(self.images):
            G_img = img.get_ker(compute_mean=self.compute_mean)
            img_kers.append(G_img)
            img_offset.append(np.concatenate([np.zeros((G_img.shape[0], i)), np.ones((G_img.shape[0], 1)), np.zeros((G_img.shape[0], len(self.images) - 1 - i))], axis=1))
            b.append(img.get_disp(self.compute_mean))
        b = np.concatenate(b)
        ker = np.concatenate(img_kers)
        offset = np.concatenate(img_offset)
        if smoothing_mat is None:
            smoothing = beta * self.S
        else:
            smoothing = beta * smoothing_mat
        aa = np.concatenate((ker, offset), axis=1)

        s_strike = np.concatenate((smoothing, np.zeros_like(smoothing)), axis=1)
        s_dip = np.concatenate((np.zeros_like(smoothing), smoothing), axis=1)
        s = np.concatenate((s_strike, s_dip), axis=0)
        bb = np.concatenate((s, np.zeros((s.shape[0], offset.shape[1]))), axis=1)

        G = np.concatenate((aa,
                            bb,))

        b = np.concatenate((b, np.zeros(s.shape[0])))
        sol = optimize.nnls(G, b)
        self.solution = sol[0]
        self.cost = sol[1]

    def solve_g(self, get_G, G_kw={}, solver='nnls', bounds=None):
        """
        solve the inversion for user defined A

        Args:
            get_G: function that build A from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {ar1:val1, ar2:val2, ... , arg_n:val_n}
        """
        b, G = get_G(self, **G_kw)
        if solver == 'nnls':
            sol = optimize.nnls(G, b)
            self.solution = sol[0]
            self.cost = sol[1]
        elif solver == 'lstsq':
            sol = np.linalg.lstsq(G, b, rcond=None)
            self.solution = sol[0]
            self.cost = sol[1]
        elif solver == 'lstsq_bound':
            if bounds is None:
                bounds = (-np.inf, np.inf)
            sol = optimize.lsq_linear(G, b, bounds=bounds)
            self.solution = sol['x']
            self.cost = sol['cost']
        else:
            raise(Exception(f'{solver} is an unauthorized solver. available solvers are: nnls, lstsq'))


    def plot_sources(self, cmap_max=None, cmap_min=None, view=(30, 225), title='Fault Geometry', I=None):
        """
        plot the fault plain

        Args:
            cmap_max(float): slip colormap maximum
            cmap_min(float): slip colormap minimum
            view(tuple): view angle (above horizon, angle)
            title(string): plot title
            set: ser of the dataset to plot with default inv.images[0]
            I: plains index to black out (debaging)

        Returns:
            fig: plot figure
            ax: plot axis

        """

        def plot_s(ax, movment=None, plot_color_bar=True, cmap_max=1.0, cmap_min=0.0, cmap='jet', title='',
                         I=-1):
            if movment is not None:

                my_cmap = cm.get_cmap(cmap)
                norm = mlb.colors.Normalize(cmap_min, cmap_max)
                shift = 0
                for p in self.plains:
                    p.plot_sources(movment[shift: shift + len(p.sources)], ax, my_cmap, norm)
                    shift += len(p.sources)
            else:
                for i, p in enumerate(self.plains):
                    t = i == I
                    p.plot_sources(movment, ax, cmap, I=t)
            ax.set_title(title)
            if movment is not None and plot_color_bar:
                cmmapable = cm.ScalarMappable(norm, my_cmap)
                cmmapable.set_array(np.linspace(cmap_min, cmap_max))
                cbar = plt.colorbar(cmmapable)
                cbar.set_label('slip [m]')

        fig = plt.figure()
        if self.solution is None:
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(*view)
            plot_s(ax, None, False, title=title, I=I)
        else:
            n = 0
            for p in self.plains:
                n += len(p.sources)
            ss = self.solution[0: n]
            ds = self.solution[n: n * 2]
            total_slip = np.sqrt(ss ** 2 + ds ** 2)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(*view)
            if cmap_max is None:
                cmax = ss.max()
            else:
                cmax = cmap_max
            if cmap_min is None:
                cmin = ss.min()
            else:
                cmin = cmap_min
            plot_s(ax, ss, cmap_max=cmax, cmap_min=cmin,  title='Strike Slip')
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(*view)
            if cmap_max is None:
                cmax = ds.max()
            else:
                cmax = cmap_max
            if cmap_min is None:
                cmin = ds.min()
            else:
                cmin = cmap_min
            plot_s(ax, ds, cmap_max=cmax, cmap_min=cmin, title='Dip Slip')
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.view_init(*view)
            # set.plot_sources(ax, total_slip, cmap_max=total_slip.max(), title='Total Slip')
            if cmap_max is None:
                cmax = total_slip.max()
            else:
                cmax = cmap_max
            if cmap_min is None:
                cmin = total_slip.min()
            else:
                cmin = cmap_min
            plot_s(ax, total_slip, cmap_max=cmax, cmap_min=cmin, title='Total Slip')
        return fig, ax


    def assign_slip(self, slip=None):
        """assign slip to each dislocation in the model """
        if slip is None:
            slip = self.solution
        n = 0
        for p in self.plains:
            n += len(p.sources)
        strike_slip = slip[0: n]
        dip_slip = slip[n: n * 2]
        shift = 0
        for i, p in enumerate(self.plains):
            p.assign_slip(strike_slip[shift: shift + len(p.sources)], dip_slip[shift: shift + len(p.sources)])
            shift += len(p.sources)






    def plot_sol(self, vmin=-0.2, vmax=0.2, cmap='jet', figsize=15, f=None, axs=None, movment=None, images=None):
        """
        plot full data model res plot

        Args:
            vmin(float): displacment colormap minimum
            vmax(float): dislpacment colormap maximum
            cmap(string): matplotlib colormap
            figsize: figure size
            f(Figure): matplotlib Figure
            axs(ndarray): matplotlib axsis array
            movment(ndarray): slip array of the shape(2n,) for n number of dislocation in the model
            images(list): subset of images to plot, default inv.images

        Returns:
            f: plot figuer
            axs: plot axis
            m_disp: model displacements (need more elaboration)

        """
        if self.solution is None:
            raise ValueError('can plot a solution only after the problem is solved')
        else:
            if images is None:
                images = self.images
            if f is None or axs is None:
                f, axs = plt.subplots(len(images), 3, figsize=(figsize * 1.5, figsize * images[0].im_y_size / images[0].im_x_size))
            n = 0
            for p in self.plains:
                n += len(p.sources)
            if movment is None:
                movment = self.solution
            strike_slip = movment[0: n]
            dip_slip = movment[n:2*n]
            m_disp = []
            if len(images) == 1:
                im, model = images[0].plot_sol(axs[0], axs[1], axs[2], strike_slip,
                                        dip_slip, self.plains, vmin, vmax, cmap, self.poisson_ratio)
                m_disp.append(model)
            else:
                for i, img in enumerate(images):
                    im, model = img.plot_sol(axs[i, 0], axs[i, 1], axs[i, 2], strike_slip,
                                            dip_slip, self.plains, vmin, vmax, cmap)
                    m_disp.append(model)
            if len(axs.shape) > 1:
                left, bottom, width, height = axs[len(images)-1, 0].get_position().bounds
            else:
                left, bottom, width, height = axs[len(images) - 1].get_position().bounds
            cax = f.add_axes([left, 0.03, 0.8, height * 0.1])
            plt.colorbar(im, orientation='horizontal', cax=cax)
            return f, axs, m_disp


    def plot_sol_val(self, G,  slip=None, vmin=-0.2, vmax=0.2, cmap='jet', figsize=15, f=None, axs=None, images=None):
        """
        plot data model res plot only for data points in the model

        Args:
            G: elastic kernel
            slip(ndarray): slip array of the shape(2n,) for n number of dislocation in the model
            vmin(float): displacment colormap minimum
            vmax(float): dislpacment colormap maximum
            cmap(string): matplotlib colormap
            figsize: figure size
            f(Figure): matplotlib Figure
            axs(ndarray): matplotlib axsis array
            images(list): subset of images to plot, default inv.images


        """

        if images is None:
            images = self.images
        if f is None or axs is None:
            f, axs = plt.subplots(len(images), 3, figsize=(figsize * 1.5, figsize * images[0].im_y_size / images[0].im_x_size))

        if slip is None:
            if self.solution is None:
                raise ValueError('can plot a solution only after the inversion solution is not None or slip is provided')
            slip = self.solution
        model = G.dot(slip)
        if len(images) == 1:
            images[0].plot_sol_val(self.plains, axs[0], axs[1], axs[2], model, vmin, vmax, cmap)
        else:
            shift = 0
            for i, img in enumerate(images):
                img.plot_sol_val(self.plains, axs[i, 0], axs[i, 1], axs[i, 2], model[shift:shift+len(img.station)], vmin, vmax, cmap)
                shift += len(img.station)
        if len(axs.shape) > 1:
            left, bottom, width, height = axs[len(images)-1, 0].get_position().bounds
        else:
            left, bottom, width, height = axs[len(images) - 1].get_position().bounds
        cax = f.add_axes([left, 0.03, 0.8, height * 0.1])
        my_cmap = cm.get_cmap(cmap)
        norm = mlb.colors.Normalize(vmin, vmax)
        cmmapable = cm.ScalarMappable(norm, my_cmap)
        cmmapable.set_array(np.linspace(vmin, vmax))
        plt.colorbar(cmmapable, orientation='horizontal', cax=cax)



    def sol_to_geojson(self, G, images=None, path='./image'):
        """
        save data model res plot data to geojson file

        Args:
            G: elasitic kernel
            images(list): subset of images to plot, default inv.images
            path(string): path to save geojson files
        """
        if self.solution is None:
            raise ValueError('can plot a solution only after the problem is solved')
        else:
            if images is None:
                images = self.images
            model = G.dot(self.solution)
            shift = 0
            for i, img in enumerate(images):
                img.sol_to_geojson(model[shift:shift+len(img.station)], path + '_%d.geojson' % i)
                shift += len(img.station)

    def plot_stations(self, dots=False, cmap='jet', vmax=1.0, vmin=-1.0, figsize=(12, 3)):
        """
        plot data points/ stations location on the dens data sets

        Args:
            dots(bool: if True plot as dots if False plot as polygons represented bt the data point
            cmap(string): matplotlib colormap
            vmax(float): colormap maximum
            vmin(float): color map minimum
            figsize(tuple): figures ize

        Returns:
            f: matplotlib figure
            axs: matplotlib axis

        """
        f, axs = plt.subplots(1, len(self.images), figsize=figsize)
        if len(self.images) == 1:
            self.images[0].plot_stations(axs, self.plains, dots, cmap, vmax=vmax, vmin=vmin)
        else:
            for i, img in enumerate(self.images):
                img.plot_stations(axs[i], self.plains, dots, cmap, vmax=vmax, vmin=vmin)
        return f, axs

    def plot_stations_val(self, cmap='jet', vmax=1.0, vmin=-1.0, figsize=(12, 3)):
        """
        plot data points/ stations with only the value of the data points

        Args:
            cmap(string): matplotlib colormap
            vmax(float): colormap maximum
            vmin(float): color map minimum
            figsize(tuple): figures ize

        Returns:
            f: matplotlib figure
            axs: matplotlib axis

        """
        f, axs = plt.subplots(1, len(self.images), figsize=figsize)
        if len(self.images) == 1:
            self.images[0].plot_stations_val(axs, self.plains, cmap, vmax=vmax, vmin=vmin)
        else:
            for i, img in enumerate(self.images):
                img.plot_stations_val(axs[i], self.plains, cmap, vmax=vmax, vmin=vmin)
        return f, axs

    def roughness2cost(self, get_G, G_kw, get_S, S_kw, betas, ax=None):
        """
        compute and plot roughness 2 plot graph

        Args:
            get_G: function that build A from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {arg1:val1, arg2:val2, ... , arg_n:val_n}
            get_S: function that build the smoothing operator from the form get_S(inv, arg1, arg2, ... arg_n)
            S_kw: map of the form {arg1:val1, arg2:val2, ... , arg_n:val_n}
            betas: list of smoothing coefficient values
            ax: an axis of shape (2,) to plot on


        """
        sourced_num = self.get_sources_num()
        S = get_S(self, *S_kw)
        total_roughness = []
        ss_roughness = []
        ds_roughness = []
        cost = []
        for beta in betas:
            G_kw['beta'] = beta
            self.solve_g(get_G, G_kw)
            sslip = self.solution[0:sourced_num]
            dslip = self.solution[sourced_num + 1:sourced_num * 2 + 1]
            slip = np.sqrt(sslip ** 2 + dslip ** 2)
            rho_slip = np.dot(S, slip)
            rho_sslip = np.dot(S, sslip)
            rho_dslip = np.dot(S, dslip)
            total_roughness.append(np.sum(np.abs(rho_slip)) / (2 * sourced_num))
            ss_roughness.append(np.sum(np.abs(rho_sslip)) / (2 * sourced_num))
            ds_roughness.append(np.sum(np.abs(rho_dslip)) / (2 * sourced_num))
            cost.append(self.cost)
            # self.plot_sources()
            # plt.show()
        if ax is None:
            fig, ax = plt.subplots(1, 2)
        ax[0].plot(total_roughness, (np.array(cost) / np.sqrt(self.get_stations_num()))*100, color='r')
        ax[1].plot(betas, (np.array(cost) / np.sqrt(self.get_stations_num())) * 100, color='r')


    def get_sources_num(self):
        """
        Returns:
            the number of sources
        """
        return self.sources_mat.shape[0]

    def get_stations_num(self):
        """

        Returns:
            the number of station / data points

        """
        s_num = 0
        for img in self.images:
            s_num += img.stations_mat.shape[0]
        return s_num
    


    def compute_cn(self):
        """
        compute the condition number of the inversion for dens data set alone with smoothing for min[Am - b] m >= 0

        Returns:
            the condition number

        """
        kernal_array = [img.get_ker(zero_pad=0, compute_mean=self.compute_mean) for img in self.images]
        G_t = np.concatenate(kernal_array)
        singular_values = np.linalg.svd(G_t, compute_uv=False)
        return singular_values.max()/singular_values.min()

    def compute_cn_g(self, get_G, G_kw):
        """
        compute the condition number of the inversion for user defined A

        Args:
            get_G: function that build A from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {ar1:val1, ar2:val2, ... , arg_n:val_n}

        Returns:
            the condition number

        """
        G = get_G(self, *G_kw)
        singular_values = np.linalg.svd(G, compute_uv=False)
        return singular_values.max()/singular_values.min()


    def resample_data(self, get_G, G_kw, min_data_size=0.2, N=5, data_per_r=1, path=None):
        """
        resample model data space while optimizing the CN

        Args:
            get_G: function that build the elastic kernal from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {ar1:val1, ar2:val2, ... , arg_n:val_n}
            min_data_size(float): minimum data point size
            N(int): number of resampling rounds
            data_per_r(int): number of data points per round
            path(int): path to save the data points of each iteration None for dont save

        Returns:

        """
        def get_cn(G):
            a = np.linalg.svd(G, compute_uv=False)
            return a.max() / a.min()
        cn_vec = []
        data_points = []
        res = []
        kernal_array = [img.get_ker(compute_mean=self.compute_mean) for img in self.images] + [seis.get_G() for seis in self.seismisity]
        G_kw['kernal_arry'] = kernal_array
        G = get_G(self, G_kw)
        cn_vec.append(get_cn(G))
        station_num = np.sum(np.array([len(img.station) for img in self.images]))
        data_points.append(station_num)
        self.build_sources_mat()
        for n in range(N):
            print('%d/%d' %(n, N))
            cn = []
            image_num = []
            ind_in_image = []
            for j, img in enumerate(self.images):
                stored_G = kernal_array[j]
                for i in range(len(img.station)):
                    # if inv.station[i].x_size/2 < inv.x_pixel or inv.station[i].y_size/2 < inv.y_pixel:
                    if img.station[i].x_size / 2 < min_data_size or img.station[i].y_size / 2 < min_data_size:
                        cn.append(1.0 * 1e99)
                        image_num.append(j)
                        ind_in_image.append(i)
                        continue
                    temp_G = img.resample(i, self.strike_element, self.dip_element, self.plains, self.compute_mean,
                                          self.poisson_ratio, self.sources_mat)
                    kernal_array[j] = temp_G
                    G_kw['kernal_arry'] = kernal_array
                    G_t = get_G(self, G_kw)
                    cn_tt = get_cn(G_t)
                    cn.append(cn_tt)
                    image_num.append(j)
                    ind_in_image.append(i)
                kernal_array[j] = stored_G
            cn_indices = np.array(cn).argsort()
            cn_indices = cn_indices[:data_per_r]
            img_indices = [[] for x in range(len(self.images))]
            for k in cn_indices:
                img_indices[image_num[k]].append(ind_in_image[k])
            for j in range(len(self.images)):
                self.images[j].add_new_stations(img_indices[j], self.strike_element, self.dip_element,
                                                self.plains, self.compute_mean)
                kernal_array[j] = self.images[j].get_ker(compute_mean=self.compute_mean)
            ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
            G_kw['kernal_arry'] = ker_array
            G = get_G(self, G_kw)
            cn_vec.append(get_cn(G))
            data_points.append(np.sum(np.array([len(img.station) for img in self.images])))
            if path is not None:
                self.save_stations_mat(path + '%d' %n)
        return data_points, cn_vec, res


    def resample_model(self, get_G, G_kw, N=5, min_size=0.2, num_of_sr=1):
        """
        resample the model space

        Args:
            get_G: function that build A from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {ar1:val1, ar2:val2, ... , arg_n:val_n}
            N(int): number of resampling rounds
            min_size: minimum dislocation size
            num_of_sr: number of dislocation to split in each round

        Returns:

        """
        def get_cn(G):
            a = np.linalg.svd(G, compute_uv=False)
            return a.max() / a.min()
        cn_vec = []
        sources_num = []
        ### building G for Ridgecrest earthquake
        ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
        G_kw['kernal_arry'] = ker_array
        G = get_G(self, G_kw)
        cn_vec.append(get_cn(G))
        self.build_sources_mat()
        # sources_num.append(self.images[0].sources_mat.shape[0])
        for n in range(N):
            print(n)
            cn = []
            plain_num = []
            source_in_plain = []
            mat_ind_vec = []
            mat_ind = -1
            for i in range(len(self.plains)):
                for s_ind in range(len(self.plains[i].sources)):
                    plain_num.append(i)
                    source_in_plain.append(s_ind)
                    mat_ind += 1
                    mat_ind_vec.append(mat_ind)
                    if self.plains[i].sources[s_ind].length < min_size or self.plains[i].sources[s_ind].width < min_size:
                        cn.append(1e99)
                        continue
                    ker_array = [img.resample_model(i, s_ind, mat_ind, self.strike_element, self.dip_element,
                                                     self.compute_mean, self.poisson_ratio, self.plains) for img in self.images] + [seis.resample_model(i, s_ind, mat_ind, self.strike_element, self.dip_element, self.plains) for seis in self.seismisity]
                    G_kw['kernal_arry'] = ker_array
                    G = get_G(self, G_kw)
                    cn.append(get_cn(G))
            cn_indices = np.array(cn).argsort()
            cn_indices = cn_indices[:num_of_sr]
            plain_num = np.array(plain_num)
            source_in_plain = np.array(source_in_plain)
            mat_ind_vec = np.array(mat_ind_vec)
            self.add_new_source(plain_num[cn_indices], source_in_plain[cn_indices], mat_ind_vec[cn_indices])
            ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
            G_kw['kernal_arry'] = ker_array
            G = get_G(self, G_kw)
            cn_vec.append(get_cn(G))
            # sources_num.append(self.images[0].sources_mat.shape[0])
        return cn_vec

    def add_new_source(self, plain_inds, sources_inds, mat_inds):
        new_sources = []
        shift_mat = 0
        for k, plain in enumerate(self.plains):
            x = np.where(plain_inds == k)[0]
            t_sources_inds = sources_inds[x]
            t_mat_inds = mat_inds[x]
            y = np.argsort(t_sources_inds)
            t_sources_inds = t_sources_inds[y]
            t_mat_inds = t_mat_inds[y]
            new_sources.append(list(plain.sources))
            shift = 0
            for i, t_mat_ind in enumerate(t_mat_inds):
                sr = plain.sources[t_sources_inds[i]]
                SR = sr.make_new_source()
                shifted = t_sources_inds[i] + shift
                shift += 3
                new_sources[k][shifted] = SR[0]
                new_sources[k].insert(shifted + 1, SR[1])
                new_sources[k].insert(shifted + 2, SR[2])
                new_sources[k].insert(shifted + 3, SR[3])
                mat_in = []
                for s in SR:
                    mat_in.append(np.array(
                        [s.length, s.width, s.depth, np.rad2deg(s.dip), np.rad2deg(s.strike), 0, 0, 0, 0, 0, s.e, s.n,
                         s.x, s.y],
                        dtype='float64'))
                mat_in = np.vstack(mat_in)
                shifted_mat = t_mat_inds[i] + shift_mat
                self.sources_mat = np.insert(np.delete(self.sources_mat, shifted_mat, axis=0), shifted_mat, mat_in,
                                             axis=0)
                for img in self.images:
                    img.insert_column(shifted_mat, SR, plain.strike_element * self.strike_element, plain.dip_element * self.dip_element)
                for seismisity in self.seismisity:
                    seismisity.insert_column(plain.strike_element * self.strike_element, plain.dip_element * self.dip_element, shifted_mat, SR)
                shift_mat += 3
        for j, plain in enumerate(self.plains):
            plain.sources = new_sources[j]



    def combine_resample(self, get_G, G_kw, final_num_of_sources, min_source_size, min_data_point_size, data_point_per_round, ratio=10, dest_path=None):
        """
        combined model and data space resampling of the inversion to optimize the condition number

        Args:
            get_G: function that build A from the form get_G(inv, arg1, arg2, ... arg_n)
            G_kw: map of the form {ar1:val1, ar2:val2, ... , arg_n:val_n}
            final_num_of_sources: exit condition finale number of desired dislocation in the model
            min_source_size: minimum dislocation size in the model
            min_data_point_size: minimum data point size
            data_point_per_round: number of data points to split in each round
            dest_path: destenation path for saving intermidate results, /None for dont save

        Returns:
            iteration, num_of_sources, num_of_stations, cn: list of the values for each iteration

        """
        def get_cn(G):
            a = np.linalg.svd(G, compute_uv=False)
            return a.max() / a.min()
        def print_status():
            print('number of station:%d, number of sources:%d, ratio%.3f' %(self.get_stations_num(), self.get_sources_num(), self.get_stations_num() / self.get_sources_num()))
        print_status()

        pre_rounds = int(np.round((self.get_stations_num() - self.get_sources_num() * ratio) / (3 * data_point_per_round)))
        if data_point_per_round > 10.0:
            rounds = int(np.round(data_point_per_round/10.0))
            data_point_per_round = 10
        else:
            rounds = 1
        iteration = []
        itert = 0
        num_of_sources = []
        num_of_stations = []
        cn = []
        num_of_sources.append(self.get_sources_num())
        num_of_stations.append(self.get_stations_num())
        ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
        G_kw['kernal_arry'] = ker_array
        G = get_G(self, G_kw)
        cn.append((get_cn(G)))
        iteration.append(0)
        if dest_path is not None:
            self.save_sources_mat(dest_path + '0')
            self.save_stations_mat(dest_path + '0')
        for i in range(pre_rounds):
            self.resample_model(get_G, G_kw, N=1, min_size=min_source_size)
            itert += 1
            if dest_path is not None:
                self.save_sources_mat(dest_path + '{}'.format(itert))
            num_of_sources.append(self.get_sources_num())
            num_of_stations.append(self.get_stations_num())
            ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
            G_kw['kernal_arry'] = ker_array
            G = get_G(self, G_kw)
            cn.append((get_cn(G)))
            iteration.append(itert)
            print_status()
        print_status()
        iter_num = int(np.round((final_num_of_sources - self.get_sources_num())/3))
        for _ in range(iter_num):
            self.resample_model(get_G, G_kw, N=1, min_size=min_source_size)
            itert += 1
            if dest_path is not None:
                self.save_sources_mat(dest_path + '{}'.format(itert))
            num_of_sources.append(self.get_sources_num())
            num_of_stations.append(self.get_stations_num())
            ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
            G_kw['kernal_arry'] = ker_array
            G = get_G(self, G_kw)
            cn.append((get_cn(G)))
            iteration.append(itert)
            print_status()
            for _ in range(rounds):
                self.resample_data(get_G, G_kw, data_per_r=data_point_per_round, N=1, min_data_size=min_data_point_size)
                itert += 1
                if dest_path is not None:
                    self.save_stations_mat(dest_path + '{}'.format(itert))
                num_of_sources.append(self.get_sources_num())
                num_of_stations.append(self.get_stations_num())
                ker_array = [img.get_ker() for img in self.images] + [seis.get_G() for seis in self.seismisity]
                G_kw['kernal_arry'] = ker_array
                G = get_G(self, G_kw)
                cn.append((get_cn(G)))
                iteration.append(itert)
                print_status()
        print_status()
        return iteration, num_of_sources, num_of_stations, cn

    def moment_magnitude(self, convert_to_meter=1e3, solution=None):
        return (np.log10(self.seismic_moment(convert_to_meter, solution)) - 9.05) / 1.5

    def seismic_moment(self, convert_to_meter=1e3, solution=None):
        if solution is None:
            solution = self.solution
        seismic_moment = 0
        mu = 30e9
        sources_num = 0
        for p in self.plains:
            sources_num += len(p.sources)
        n = 0
        for plain in self.plains:
            strike_slip = solution[n: n + len(plain.sources)]
            dip_slip = solution[n + sources_num: n + sources_num + len(plain.sources)]
            n += len(plain.sources)
            seismic_moment += plain.seismic_moment(strike_slip, dip_slip, convert_to_meter, mu)
        return seismic_moment
    
    def part_seismic_moment(self, solution, convert_to_meter=1e3, plains=None):
        seismic_moment = 0
        mu = 30e9
        sources_num = 0
        for p in self.images[0].plains:
            sources_num += len(p.sources)
        shift = 0
        if plains is None:
            plains = range(len(self.images[0].plains))
        for j, plain in enumerate(self.images[0].plains):
            if j in plains:
                movment = solution[shift: shift+len(plain.sources)]
                for i, sr in enumerate(plain.sources):
                    seismic_moment += sr.length * convert_to_meter * sr.width * convert_to_meter * mu * movment[i]
            shift += len(plain.sources)
        return seismic_moment



    def quadtree(self,  threshold, min_size):
        if type(threshold) is not float > 1:
            assert len(threshold) == len(min_size), 'threshold length is %d and min size is %d shold by the same' %(len(threshold), len(min_size))
            for th, mi, img in zip(threshold, min_size, self.images):
                img.quadtree(th, mi, 0.2)
        else:
            for img in self.images:
                img.quadtree(threshold, min_size, 0.2)


    def solve_non_linear_torch(self, strike, dip, ss, ds, length, width, e, n, depth):
        import torch
        from disloc_torch import disloc_pytorch
        model = torch.autograd.Variable(torch.DoubleTensor([strike, dip, ss, ds, length, width, e, n, depth]),
                                        requires_grad=True)
        depth = model[-1] + torch.sin(model[1]) * model[5]



        east0 = torch.DoubleTensor(self.images[0].stations_mat[:, 0]) - model[6]
        north0 = torch.DoubleTensor(self.images[0].stations_mat[:, 1]) - model[7]
        b0 = torch.DoubleTensor(self.images[0].stations_mat[:, 4])

        east1 = torch.DoubleTensor(self.images[1].stations_mat[:, 0]) - model[6]
        north1 = torch.DoubleTensor(self.images[1].stations_mat[:, 1]) - model[7]
        b1 = torch.DoubleTensor(self.images[1].stations_mat[:, 4])

        strike_ker = disloc_pytorch(model[4], model[5], depth, model[1], model[0], 0.0, 0.0,
                                    1.0,
                                    0.0, 0.0, east0, north0, 0.25)

        x0 = -np.cos(self.images[0].azimuth) * strike_ker[0]
        y0 = np.sin(self.images[0].azimuth) * strike_ker[1]
        z0 = strike_ker[2] * np.cos(self.images[0].incidence_angle)

        strike_ker0 = -((x0 + y0) * np.sin(self.images[0].incidence_angle) + z0)

        dip_ker = disloc_pytorch(model[4], model[5], depth, model[1], model[0], 0.0, 0.0, 0.0,
                                 1.0, 0.0, east0, north0, 0.25)

        x0 = -np.cos(self.images[0].azimuth) * dip_ker[0]
        y0 = np.sin(self.images[0].azimuth) * dip_ker[1]
        z0 = dip_ker[2] * np.cos(self.images[0].incidence_angle)
        dip_ker0 = -((x0 + y0) * np.sin(self.images[0].incidence_angle) + z0)



        strike_ker = disloc_pytorch(model[4], model[5], depth, model[1], model[0], 0.0, 0.0, 1.0,
                             0.0, 0.0, east1, north1, 0.25)

        x1 = -np.cos(self.images[1].azimuth) * strike_ker[0]
        y1 = np.sin(self.images[1].azimuth) * strike_ker[1]
        z1 = strike_ker[2] * np.cos(self.images[1].incidence_angle)
        strike_ker1 = -((x1 + y1) * np.sin(self.images[1].incidence_angle) + z1)



        dip_ker = disloc_pytorch(model[4], model[5], depth, model[1], model[0], 0.0, 0.0, 0.0,
                             1.0, 0.0, east1, north1, 0.25)
        x1 = -np.cos(self.images[1].azimuth) * dip_ker[0]
        y1 = np.sin(self.images[1].azimuth) * dip_ker[1]
        z1 = dip_ker[2] * np.cos(self.images[1].incidence_angle)
        dip_ker1 = -((x1 + y1) * np.sin(self.images[0].incidence_angle) + z1)

        strike_ker = torch.cat((strike_ker0, strike_ker1))
        dip_ker = torch.cat((dip_ker0, dip_ker1))
        b = torch.cat((b0, b1))



    # img_kers.append(torch.stack((strike_ker * w, dip_ker * w)))

        ker = torch.stack((strike_ker , dip_ker )).t()
        norm = torch.norm(torch.mm(ker, model[2:4].view(-1, 1)).view(-1) - b, 2)
        norm.backward()
        return norm.detach().numpy(), model.grad.numpy()



    def uncorelated_noise(self, mus, sigmas):
        return [img.uncorelated_noise(mu, sigma) for img, mu, sigma in zip(self.images, mus, sigmas)]

    def corelated_noise(self, sigma, max_noise, num_of_points):
            return [img.corelated_noise(s, ms, nop) for img , s, ms, nop in zip(self.images, sigma, max_noise, num_of_points)]

    def restor_clean_displacment(self):
        for img in self.images:
            img.restore_clean_displacment()

    def save_stations_mat(self, pref):
        for i, img in enumerate(self.images):
            np.save(pref + '_image_%d.npy' % i, img.stations_mat)

    def calc_coulomb(self, mu, X, Y, Z, strike=0.0, dip=(3.141592653589793 / 2.0), rake=0.0, lambda_l =50e9, shear_m=30e9):
        def normal(strike, dip):
            nz = np.cos(dip)
            nx = np.cos(strike) * np.sin(dip)
            ny = -np.sin(strike) * np.sin(dip)
            return np.array([nx, ny, nz]).reshape(-1, 1)

        def shear_hat(strike, dip, rake):
            l_s = np.cos(rake)
            l_d = np.sin(rake)
            nz = -np.sin(dip) * l_d
            l = np.cos(dip) * l_d

            x_s = np.cos(strike) * l_s
            y_s = np.sin(strike) * l_s

            x_d = np.cos(strike + np.pi / 2) * l
            y_d = np.sin(strike + np.pi / 2) * l
            return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)

        self.assign_slip()

        stress = np.zeros((X.shape[0], 3, 3))

        for plain in self.plains:
            for sr in plain.sources:
                if sr.strike_slip < 1e-7 and sr.dip_slip < 1e-7:
                    continue
                for ix in range(X.shape[0]):
                    stress[ix] += sr.stress(X[ix], Y[ix], Z[ix], self.strike_element*plain.strike_element, self.dip_element*plain.dip_element, lambda_l, shear_m)
        n_hat = normal(strike, dip)
        s_hat = shear_hat(np.pi / 2 - strike, dip, rake)
        t = np.squeeze(stress.dot(n_hat))
        tn = -np.squeeze(t.dot(n_hat))
        ts = np.squeeze(t.dot(s_hat))
        coulomb = ts + mu * tn
        return X, Y, Z, coulomb, tn, ts

    def calc_coulomb_disloc(self, mu, X, Y, Z, strike, dip, rake, lambda_l=50e9, shear_m=30e9):
        def normal(strike, dip):
            nz = np.cos(dip)
            nx = np.cos(strike) * np.sin(dip)
            ny = -np.sin(strike ) * np.sin(dip)
            return np.array([nx, ny, nz]).reshape(-1, 1)

        def shear_hat(strike, dip, rake):
            l_s = np.cos(rake)
            l_d = np.sin(rake)
            nz = -np.sin(dip) * l_d
            l = np.cos(dip) * l_d

            x_s = np.cos(strike) * l_s
            y_s = np.sin(strike) * l_s

            x_d = np.cos(strike + np.pi / 2) * l
            y_d = np.sin(strike + np.pi / 2) * l
            return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)

        self.assign_slip()

        tn = []
        ts = []
        coulomb = []
        for ix in range(X.shape[0]):
            stress = np.zeros((3, 3))
            for plain in self.images[0].plains:
                for isr, sr in enumerate(plain.sources):
                    if (sr.strike_slip < 1e-7 and sr.dip_slip < 1e-7):
                        continue
                    stress += sr.stress(X[ix], Y[ix], Z[ix], self.strike_element*plain.strike_element, self.dip_element*plain.dip_element, lambda_l, shear_m)
            n_hat = normal(strike[ix], dip[ix])
            s_hat = shear_hat(np.pi / 2 - strike[ix], dip[ix], rake[ix])
            t = np.squeeze(stress.dot(n_hat))
            tn.append(np.squeeze(t.dot(n_hat)))
            ts.append(np.squeeze(t.dot(s_hat)))
            coulomb.append(ts[-1] - mu * tn[-1])
        return np.array(coulomb), np.array(tn), np.array(ts)

    def calc_coulomb_cord(self, mu, x, y, z, strike, dip, rake, lambda_l=50e9, shear_m=30e9):
        def normal(strike, dip):
            nz = np.cos(dip)
            nx = np.cos(strike) * np.sin(dip)
            ny = -np.sin(strike ) * np.sin(dip)
            return np.array([nx, ny, nz]).reshape(-1, 1)

        def shear_hat(strike, dip, rake):
            l_s = np.cos(rake)
            l_d = np.sin(rake)
            nz = -np.sin(dip) * l_d
            l = np.cos(dip) * l_d

            x_s = np.cos(strike) * l_s
            y_s = np.sin(strike) * l_s

            x_d = np.cos(strike + np.pi / 2) * l
            y_d = np.sin(strike + np.pi / 2) * l
            return np.array([x_s + x_d, y_s + y_d, nz]).reshape(-1, 1)

        self.assign_slip()


        stress = np.zeros((3, 3))
        for plain in self.images[0].plains:
            for isr, sr in enumerate(plain.sources):
                if (sr.strike_slip < 1e-7 and sr.dip_slip < 1e-7):
                    continue
                stress += sr.stress(x, y, z, self.strike_element*plain.strike_element, self.dip_element*plain.dip_element, lambda_l, shear_m)
        n_hat = normal(strike, dip)
        s_hat = shear_hat(np.pi / 2 - strike, dip, rake)
        t = np.squeeze(stress.dot(n_hat))
        tn = np.squeeze(t.dot(n_hat))
        ts = np.squeeze(t.dot(s_hat))
        coulomb = ts - mu * tn
        return tn, ts, coulomb


    def calc_plains_colum(self, ax, spacing=1.0, lambda_l=30e9, shear_m=30e9, mu=0.6, rake=0.0, cmap='jet', vmin=1e9, vmax=1e10):

        my_cmap = mlb.cm.get_cmap(cmap)
        norm = mlb.colors.Normalize(vmin, vmax)

        for plain in self.images[0].plains:
            X_p, Y_p = np.meshgrid(np.arange(0, plain.plain_length, spacing), np.arange(0, plain.total_width, spacing))
            strike = np.deg2rad(plain.strike)
            ccw_to_x_stk = np.pi / 2 - strike  # the angle betuen the fualt and the x axis cunter clock wise
            ccw_to_x_dip = - strike
            dip = np.deg2rad(plain.dip-0.1)
            e = plain.plain_cord[0] + X_p * np.cos(ccw_to_x_stk) + np.cos(ccw_to_x_dip) * np.cos(dip) * Y_p
            n = np.ones_like(e) * 50.0#plain.plain_cord[1] + X_p * np.sin(ccw_to_x_stk) + np.sin(ccw_to_x_dip) * np.cos(dip) * Y_p
            depth = (np.abs(plain.plain_cord[2]) + np.sin(dip) * Y_p)
            X, Y, Z, coulomb, tn, ts = self.calc_ccoulomb(mu, e.flatten(), n.flatten(), depth.flatten(), strike=np.deg2rad(plain.strike), dip=np.deg2rad(plain.dip), rake=np.deg2rad(rake), lambda_l=lambda_l, shear_m=shear_m)
            ax.plot_surface(e, n, depth, facecolors=my_cmap(norm(ts.reshape(n.shape))))
            ax.set_ylim(40,60)
            print(coulomb)
        cmmapable = plt.cm.ScalarMappable(norm, my_cmap)
        cmmapable.set_array(np.linspace(vmin, vmax))
        cbar = plt.colorbar(cmmapable)
        plt.figure()
        sum = []
        A = np.linspace(ts.min(), ts.max(), 100)
        for a in A:
            sum.append(np.sum(ts < a) / float(ts.shape[0]))
        plt.plot(A, sum)
        plt.xscale('symlog')


    def slip_depth(self, max_depth=15, intervals=10):
        self.assign_slip()
        plt.figure()
        depth = []
        slip = []
        length = []
        for p in self.images[0].plains:
            for s in p.sources:
                depth.append(s.depth_m)
                slip.append(np.sqrt(s.strike_slip**2 +s.dip_slip**2))
                length.append(s.length)
        depth = np.array(depth)
        slip = np.array(slip)
        length = np.array(length)
        d_depth = []
        d_slip = []
        intrtv = np.linspace(0, max_depth, intervals)
        for i, d in enumerate(intrtv[:-1]):
            inds = np.argwhere(np.logical_and(depth >= d, depth < intrtv[i+1]))
            if inds.shape[0] != 0:
                d_depth.append(np.mean(depth[inds]))
                d_slip.append(np.sum((slip[inds].flatten() * length[inds].flatten())) / np.sum(length[inds]))
        plt.plot(d_slip, d_depth)
        plt.ylim((max_depth, 0))
        plt.axis('scaled')
        plt.xlim((0,3))


    def plains_to_qgis(self, path):
        wkt = []
        for p in self.plains:
            x, y = p.get_fault(1.0, 1.0, 2)
            wkt.append('LINESTRING (%f %f, %f %f)' %(x[0], y[0], x[1], y[1]))
        with open(path, 'w') as f:
            f.write('\n'.join(wkt))

    def build_sources_mat(self):
        mat = []
        for plain in self.plains:
            for sr in plain.sources:
                mat.append(np.array(
                    [sr.length, sr.width, sr.depth, np.rad2deg(sr.dip), np.rad2deg(sr.strike), 0, 0, 0, 0, 0, sr.e, sr.n
                        , sr.x, sr.y],
                    dtype='float64'))
        self.sources_mat = np.vstack(mat)

    def save_sources_mat(self, file_prefix):
        assert self.sources_mat is not None, 'need to initialize sources_mat before saving it'
        n = 0
        for i, plain in enumerate(self.plains):
            np.save(file_prefix + '_plain_{}.npy'.format(i), self.sources_mat[n: n + len(plain.sources), :])
            n += len(plain.sources)

    def to_gmt(self, path, slip, plains=None):
        if plains is None:
            plains = range(len(self.plains))
        gmt_file = ''
        i=0
        for ip, p in enumerate(self.plains):
            if ip in plains:
                for s in p.sources:
                    gmt_file += s.to_gmt(slip[i])
                    i += 1
            else:
                i += len(p.sources)
        with open(path, 'w') as f:
            f.write(gmt_file)

    def get_dislocation_centers(self):
        x = []
        y = []
        z = []
        strike = []
        dip = []
        rake = []
        for p in self.plains:
            for s in p.sources:
                x.append(s.e_m)
                y.append(s.n_m)
                z.append(s.depth_m)
                strike.append(s.strike)
                dip.append(s.dip)
                if p.strike_element == -1:
                    rake.append(np.deg2rad(0))
                elif p.strike_element == 1:
                    rake.append(np.deg2rad(180))

        return np.array(x), np.array(y), np.array(z), np.array(strike), np.array(dip), np.array(rake)

    def new_smoothing(self):
        self.build_sources_mat()
        S = np.zeros((self.sources_mat.shape[0], self.sources_mat.shape[0]))
        i = 0
        j = 0
        for ip, p in enumerate(self.plains):
            for s in p.sources:
                points = s.get_corners()
                for itp, t_p in enumerate(self.plains):
                    if ip in t_p.dont_smooth or itp in p.dont_smooth:
                        j += len(t_p.sources)
                        continue
                    for t_s in t_p.sources:
                        if(i == j):
                            j += 1
                            continue
                        tpoints = t_s.get_corners()
                        if neighbors(points[0], points[1], points[2], points[3], tpoints[0], tpoints[1], tpoints[2], tpoints[3]):
                            S[i, j] += 1.0
                            S[i, i] -= 1.0
                        j += 1
                j = 0
                i += 1
        return S

    def get_fault(self, sampels=8):
        X = []
        Y = []
        for p in self.plains:
            x, y = p.get_fault(1.0, 1.0, sampels)
            X.append(x)
            Y.append(y)
        return X, Y

