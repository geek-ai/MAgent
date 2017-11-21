from abc import ABCMeta, abstractmethod


class BaseServer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_group_info(self):
        pass

    @abstractmethod
    def get_static_info(self):
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

