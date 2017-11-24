""" base class for environment """


class Environment:
    """see subclass for detailed comment"""
    def __init__(self):
        pass

    def reset(self):
        pass

    # ====== RUN ======
    def get_observation(self, handle):
        pass

    def set_action(self, handle, actions):
        pass

    def step(self):
        pass

    def render(self):
        pass

    def render_next_file(self):
        pass

    def get_reward(self, handle):
        pass

    # ====== INFO ======
    def get_num(self, handle):
        pass

    def get_action_space(self, handle):
        pass

    def get_view_space(self, handle):
        pass

    def get_feature_space(self, handle):
        pass

