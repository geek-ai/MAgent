from .base_server import BaseServer


class SampleServer(BaseServer):
    def get_group_info(self):
        return [[1, 1, 0, 0, 0]]

    def get_static_info(self):
        return {"walls": []}

    def get_data(self, frame_id, x_range, y_range):
        if frame_id == 0:
            return {1: [10, 10, 0]}, [(1, 0, 0)]
        elif frame_id == 1:
            return {1: [9, 10, 0]}, [(1, 0, 0)]
        elif frame_id == 2:
            return {1: [8, 10, 0]}, [(1, 0, 0)]
        elif frame_id == 3:
            return {1: [14, 12, 0]}, [(1, 0, 0)]
        else:
            return {1: [10, 10, 0]}, [(1, 0, 0)]

    def add_agents(self, x, y, g):
        pass

    def get_map_size(self):
        return [50, 50]
