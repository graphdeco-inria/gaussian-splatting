class EMATracker:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema_values = {}

    def update(self, key, value):
        if key not in self.ema_values:
            self.ema_values[key] = value
        else:
            self.ema_values[key] = self.alpha * value + (1 - self.alpha) * self.ema_values[key]
        return self.ema_values[key]

    def get(self, key):
        return self.ema_values.get(key, None)