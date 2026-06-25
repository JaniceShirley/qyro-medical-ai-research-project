class EarlyStopping:
    """
    Early stopping helper to halt training when monitored validation metric
    fails to improve after a set number of epochs (patience).
    """
    def __init__(self, patience=8, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == "min":
            self.monitor_op = lambda current, best: current < (best - min_delta)
        elif mode == "max":
            self.monitor_op = lambda current, best: current > (best + min_delta)
        else:
            raise ValueError(f"Invalid early stopping mode: {mode}")

    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
            self.counter = 0
            return False
            
        if self.monitor_op(val_metric, self.best_score):
            self.best_score = val_metric
            self.counter = 0
            print(f"Validation metric improved. Resetting early stopping counter.")
            return False
        else:
            self.counter += 1
            print(f"Early stopping counter: {self.counter} out of {self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered. Halting training.")
            return self.early_stop
