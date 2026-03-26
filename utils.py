#Any extra functionality that need to be reused will go here
class _TargetData:
    def __init__(self, split: dict):
        self.X_train = split['X_train']
        self.X_test  = split['X_test']
        self.y_train = split['y_train']
        self.y_test  = split['y_test']