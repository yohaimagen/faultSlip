from copy import deepcopy





class Station:
    def __init__(self, east, north, x_size=1, y_size=1, disp=0, weight=1, flag=False):
        if flag:
            self.east = east
            self.north = north
        else:
            self.east = east + x_size / 2.0
            self.north = north + y_size / 2.0
        self.disp = disp
        self.x_size = x_size
        self.y_size = y_size
        self.weight = weight

    def make_new_stations(self):
        dx = self.x_size / 4.0
        dy = self.y_size / 4.0
        s1 = deepcopy(self)
        s1.x_size /= 2
        s1.y_size /= 2
        s1.east -= dx
        s1.north -= dy
        s2 = deepcopy(self)
        s2.x_size /= 2
        s2.y_size /= 2
        s2.east += dx
        s2.north -= dy
        s3 = deepcopy(self)
        s3.x_size /= 2
        s3.y_size /= 2
        s3.east -= dx
        s3.north += dy
        s4 = deepcopy(self)
        s4.x_size /= 2
        s4.y_size /= 2
        s4.east += dx
        s4.north += dy
        return s1, s2, s3, s4