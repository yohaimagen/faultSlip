import matplotlib.patches as patches
from matplotlib.pylab import *

from faultSlip.disloc import *
from faultSlip.source import Source


class Plain:
    def __init__(
        self,
        dip=None,
        strike=None,
        plain_cord=None,
        plain_length=None,
        width=None,
        num_sub_stk=None,
        dont_smooth=[],
        smooth_down=True,
        sources_file=False,
        total_width=None,
        strike_element=1,
        dip_element=1,
        open_element=0,
        **kwargs
    ):
        self.dip = dip
        self.strike = strike
        self.plain_cord = plain_cord
        self.plain_length = plain_length
        self.sources = []
        self.total_width = 0
        self.dont_smooth = dont_smooth
        self.smooth_down = smooth_down
        self.num_sub_stk = num_sub_stk
        self.strike_element = strike_element
        self.dip_element = dip_element
        self.open_element = open_element
        if not sources_file:
            for i in range(len(width)):
                source_length = plain_length / num_sub_stk[i]
                self.total_width += width[i]
                for j in range(int(num_sub_stk[i])):
                    self.sources.append(
                        Source(
                            None,
                            None,
                            np.deg2rad(strike),
                            np.deg2rad(dip),
                            source_length,
                            width[i],
                            0,
                            0,
                            0,
                            source_length * j + source_length / 2.0,
                            self.total_width,
                            plain_cord[0],
                            plain_cord[1],
                            plain_cord[2],
                        )
                    )
        else:
            sr_mat = np.load(sources_file)
            for i in range(sr_mat.shape[0]):
                self.sources.append(
                    Source(
                        None,
                        None,
                        np.deg2rad(sr_mat[i, 4]),
                        np.deg2rad(sr_mat[i, 3]),
                        sr_mat[i, 0],
                        sr_mat[i, 1],
                        0,
                        0,
                        0,
                        sr_mat[i, 12],
                        sr_mat[i, 13],
                        east=sr_mat[i, 10],
                        north=sr_mat[i, 11],
                        depth=sr_mat[i, 2],
                    )
                )
            self.total_width = total_width

    def get_mesh(self, sub_dip, sub_strike):
        ccw_to_x_stk = np.pi / 2 - np.deg2rad(
            self.strike
        )  # the angle betuen the fualt and the x axis cunter clock wise
        ccw_to_x_dip = -np.deg2rad(self.strike)
        dip = np.deg2rad(self.dip)
        strike_steps = int(self.plain_length / sub_strike)
        dip_steps = int(self.total_width / sub_dip)
        dx = np.linspace(
            sub_strike / 2.0,
            self.plain_length + sub_strike / 2.0,
            strike_steps,
            endpoint=False,
        )
        dy = np.linspace(
            sub_dip / 2.0, self.total_width + sub_dip / 2.0, dip_steps, endpoint=False
        )
        dX, dY = np.meshgrid(dx, dy)
        x = (
            self.plain_cord[0]
            + dX * np.cos(ccw_to_x_stk)
            + np.cos(ccw_to_x_dip) * np.cos(dip) * dY
        )
        y = (
            self.plain_cord[1]
            + dX * np.sin(ccw_to_x_stk)
            + np.sin(ccw_to_x_dip) * np.cos(dip) * dY
        )
        z = self.plain_cord[2] + np.sin(dip) * dY

        return x, y, z

    def sample_points(self, N):
        ccw_to_x_stk = np.pi / 2 - np.deg2rad(
            self.strike
        )  # the angle betuen the fualt and the x axis cunter clock wise
        ccw_to_x_dip = -np.deg2rad(self.strike)
        dip = np.deg2rad(self.dip)
        strike_step = np.random.uniform(0, self.plain_length, N)
        dip_step = np.random.uniform(0, self.total_width, N)
        x = (
            self.plain_cord[0]
            + strike_step * np.cos(ccw_to_x_stk)
            + np.cos(ccw_to_x_dip) * np.cos(dip) * dip_step
        )
        y = (
            self.plain_cord[1]
            + strike_step * np.sin(ccw_to_x_stk)
            + np.sin(ccw_to_x_dip) * np.cos(dip) * dip_step
        )
        z = self.plain_cord[2] + np.sin(dip) * dip_step
        return np.stack((x, y, z)).T

    def get_strike_ker(self, zero_pad=0):
        # adding zeros column in the end of the ss matrix
        if zero_pad != 0:
            strike_kernal_temp = np.zeros(
                (self.strike_kernal.shape[0], self.strike_kernal.shape[1] + zero_pad)
            )
            strike_kernal_temp[:, :-zero_pad] = self.strike_kernal
            return strike_kernal_temp
        return self.strike_kernal

    def get_dip_ker(self, zero_pad=0):
        # adding zeros column in the end of the ds matrix
        if zero_pad != 0:
            dip_kernal_temp = np.zeros(
                (self.dip_kernal.shape[0], self.dip_kernal.shape[1] + zero_pad)
            )
            dip_kernal_temp[:, :-zero_pad] = self.dip_kernal
            return dip_kernal_temp
        return self.dip_kernal

    def get_smothing_no_bounds(self):
        """
        build a smoothing matrix for the plain model defend by self
        :return: a tuple of the smoothing matrix, a list of the indices of the most right sources and left sources
        """

        def adjacent(sp1, sp2):
            def eq(x, y):
                return np.abs(x - y) < 1e-10

            def biger(x, y):
                if eq(x, y):
                    return False
                return x > y

            x1_left = sp1.x - 0.5 * sp1.length
            x1_right = x1_left + sp1.length
            x2_left = sp2.x - 0.5 * sp2.length
            x2_right = x2_left + sp2.length
            y1_boutom = sp1.y
            y1_up = y1_boutom - sp1.width
            y2_boutom = sp2.y
            y2_up = y2_boutom - sp2.width
            if sp1 == sp2:
                return False
            elif biger(x1_left, x2_right) or biger(x2_left, x1_right):
                return False
            elif biger(y2_up, y1_boutom) or biger(y1_up, y2_boutom):
                return False
            elif (
                (eq(x1_left, x2_right) and eq(y1_boutom, y2_up))
                or (eq(x1_left, x2_right) and eq(y1_up, y2_boutom))
                or (eq(x1_right, x2_left) and eq(y1_boutom, y2_up))
                or (eq(x1_right, x2_left) and eq(y1_up, y2_boutom))
            ):
                return False
            else:
                return True

        S = np.zeros((len(self.sources), len(self.sources)))
        left_x = np.array(
            [s_plain.x - 0.5 * s_plain.length for s_plain in self.sources]
        )
        right_x = np.array(
            [s_plain.x + 0.5 * s_plain.length for s_plain in self.sources]
        )
        min_x_ind = np.where(left_x == left_x.min())[0]
        max_x_ind = np.where(right_x == right_x.max())[0]
        # constrained eace source to be similar to it's neighbors
        for k in range(len(self.sources)):
            for j in range(len(self.sources)):
                if adjacent(self.sources[k], self.sources[j]):
                    S[k, k] += 1
                    S[k, j] -= 1
        return S, max_x_ind, min_x_ind

    def plot_sources(self, movment, ax, my_cmap=None, norm=None, I=False):
        for i, sr in enumerate(self.sources):
            if movment is None:
                if I:
                    color_sub_plain = "blue"
                else:
                    color_sub_plain = "white"
            else:
                color_sub_plain = my_cmap(norm(movment[i]))
            ccw_to_x_stk = (
                np.pi / 2 - sr.strike
            )  # the angle betuen the fualt and the x axis cunter clock wise
            ccw_to_x_dip = -sr.strike
            e1 = sr.e + sr.length / 2.0 * np.cos(ccw_to_x_stk)
            n1 = sr.n + sr.length / 2.0 * np.sin(ccw_to_x_stk)
            e2 = sr.e - sr.length / 2.0 * np.cos(ccw_to_x_stk)
            n2 = sr.n - sr.length / 2.0 * np.sin(ccw_to_x_stk)
            z1 = -sr.depth
            z2 = z1 + sr.width * np.sin(sr.dip)
            l = sr.width * np.cos(sr.dip)
            e3 = e1 - l * np.cos(ccw_to_x_dip)
            n3 = n1 - l * np.sin(ccw_to_x_dip)
            e4 = e2 - l * np.cos(ccw_to_x_dip)
            n4 = n2 - l * np.sin(ccw_to_x_dip)
            ax.plot_surface(
                np.array([[e1, e2], [e3, e4]]),
                np.array([[n1, n2], [n3, n4]]),
                np.array([[z1, z1], [z2, z2]]),
                color=color_sub_plain,
                edgecolor="black",
            )

    def plot_sources_2d(self, ax, movment, my_cmap=None, norm=None, shift=0):
        for i, sr in enumerate(self.sources):
            if movment is None:
                color_sub_plain = "gray"
            else:
                color_sub_plain = my_cmap(norm(movment[i]))
            rect = patches.Rectangle(
                (shift + sr.x - sr.length / 2, -sr.y),
                sr.length,
                sr.width,
                edgecolor="k",
                facecolor=color_sub_plain,
                zorder=1,
            )
            ax.add_patch(rect)
        # ax.set_xlim(0, self.plain_length)
        # ax.set_ylim(-self.total_width, 0)

    def assign_slip(self, strike_slip, dip_slip):
        for i, sr in enumerate(self.sources):
            sr.strike_slip = strike_slip[i]
            sr.dip_slip = dip_slip[i]

    def assign_del_slip(self, del_dslip):
        for i, sr in enumerate(self.sources):
            sr.d_slip = del_dslip[i]

    def get_fault(self, x_pixel, y_pixel, sampels=8):
        m = np.tan(np.deg2rad(450 - self.strike))
        n = self.plain_cord[1] - m * self.plain_cord[0]
        x_start = self.plain_cord[0]
        x_end = x_start + np.cos(np.deg2rad(450 - self.strike)) * self.plain_length
        if x_start > x_end:
            x_start, x_end = x_end, x_start
        x = np.linspace(x_start, x_end, sampels)
        return x / x_pixel, (m * x + n) / y_pixel

    def compute_station_disp(self, s, ss, ds, azimuth, incidence_angle, zero_padd=0):
        sources_num = len(self.sources)
        uE = np.zeros(sources_num + zero_padd, dtype="float64")
        uN = np.zeros(sources_num + zero_padd, dtype="float64")
        uZ = np.zeros(sources_num + zero_padd, dtype="float64")
        i = 0
        for sr in self.sources:
            e = np.zeros(1)
            n = np.zeros(1)
            z = np.zeros(1)
            model = np.array(
                [
                    sr.length,
                    sr.width,
                    sr.depth,
                    np.rad2deg(sr.dip),
                    np.rad2deg(sr.strike),
                    0,
                    0,
                    ss * self.strike_element,
                    ds * self.dip_element,
                    0.0,
                ],
                dtype="float64",
            )
            disloc.disloc_1d(
                e,
                n,
                z,
                model,
                np.array([s.east - sr.e]),
                np.array([s.north - sr.n]),
                0.25,
                1,
                1,
            )
            uE[i] = e[0]
            uN[i] = n[0]
            uZ[i] = z[0]
            i += 1
        return -(
            (-np.cos(azimuth) * uE + np.sin(azimuth) * uN) * np.sin(incidence_angle)
            + uZ * np.cos(incidence_angle)
        )

    def seismic_moment(self, strike_slip, dip_slip, convert_to_meter=1e3, mu=30e9):
        seismic_moment = 0
        for i, sr in enumerate(self.sources):
            seismic_moment += (
                sr.length
                * convert_to_meter
                * sr.width
                * convert_to_meter
                * mu
                * np.sqrt(strike_slip[i] ** 2 + dip_slip[i] ** 2)
            )
        return seismic_moment
