import random

from .base_server import BaseServer


class RandomServer(BaseServer):
    def __init__(self, agent_number=1000, group_number=20, map_size=100, shape_range=3, speed=5, event_range=100):
        self._data = {}
        self._map_size = map_size
        self._number = agent_number
        for i in range(agent_number):
            self._data.setdefault(i, [
                random.randint(0, map_size - 1),
                random.randint(0, map_size - 1),
                random.randint(0, group_number - 1)
            ])
        self._group = []
        for i in range(group_number):
            self._group.append([
                random.randint(1, shape_range),
                random.randint(1, shape_range),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            ])
        self._speed = speed
        self._event_range = event_range
        self._map_size = map_size

    def get_group_info(self):
        return self._group

    def get_static_info(self):
        return {"wall": []}

    def get_data(self, frame_id, x_range, y_range):
        result = {}
        event = []
        for i in self._data:
            olddata = self._data[i]
            data = [0, 0, 0]
            data[0] = olddata[0] + random.randint(-self._speed, self._speed)
            data[1] = olddata[1] + random.randint(-self._speed, self._speed)
            data[0] = min(max(data[0], 0), self._map_size - 1)
            data[1] = min(max(data[1], 0), self._map_size - 1)
            data[2] = olddata[2]
            self._data[i] = data
            if (x_range[0] <= data[0] <= x_range[1] and y_range[0] <= data[1] <= y_range[1]) or \
                    (x_range[0] <= olddata[0] <= x_range[1] and y_range[0] <= olddata[1] <= y_range[1]):
                result.setdefault(i, olddata)
        event_number = random.randint(0, self._event_range)
        for i in range(event_number):
            agent_id, _ = random.choice(self._data.items())
            event.append(
                (
                    agent_id,
                    random.randint(0, self._map_size - 1),
                    random.randint(0, self._map_size - 1)
                )
            )
        return result, event

    def add_agents(self, x, y, g):
        self._data.setdefault(self._number, (x, y, g))
        self._number += 1
        
    def get_map_size(self):
        return self._map_size, self._map_size

