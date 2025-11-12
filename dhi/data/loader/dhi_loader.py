from dhi.data.loader._base import Loader

class DHILoader(Loader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load(self):
        # NOTE: at the moment of writing this class, no extra-loading steps are needed
        return super().load()