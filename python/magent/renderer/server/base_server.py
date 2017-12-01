from abc import ABCMeta, abstractmethod


class BaseServer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def get_data(self, frame_id, x_range, y_range):
        pass

    @abstractmethod
    def add_agents(self, x, y, g):
        pass
        
    @abstractmethod
    def get_map_size(self):
        pass

    @abstractmethod
    def get_banners(self, frame_id, resolution):
        pass

    @abstractmethod
    def get_status(self, frame_id):
        pass

    @abstractmethod
    def keydown(self, frame_id, key, mouse_x, mouse_y):
        pass

    @abstractmethod
    def mousedown(self, frame_id, key, mouse_x, mouse_y):
        pass

    @abstractmethod
    def get_endscreen(self, frame_id):
        pass