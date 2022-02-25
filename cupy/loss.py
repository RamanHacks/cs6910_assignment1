import cupy as cp

class Loss:
    def __init__(self, prediction=None, target=None):
        self.prediction = prediction
        self.target = target
        self.eps = 1e-6

    def forward(self, p, t):
        pass

    def backward(self, p, t):
        pass
    
    def __call__(self, p, t):
        return self.forward(p, t)

class MSE(Loss):
    def __init__(self, prediction=None, target=None):
        super().__init__(prediction, target)
    def forward(self, p, t):
        return {'loss':cp.mean((p - t) ** 2, axis=0) / 2, 'grad':self.backward()}

    def backward(self):
        return self.prediction - self.target


class CrossEntropy(Loss):
    def __init__(self, prediction=None, target=None):
        super().__init__(prediction, target)
    #  https://cs231n.github.io/neural-networks-case-study/#grad
    
    def forward(self, p, t):
        self.prediction = p
        self.target = t
        loss = cp.sum(-t*self.logsoftmax(p),axis=-1)

        return {'loss':cp.mean(loss, axis=0), 'grad':self.backward()}

    def backward(self):
        #  https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/
        probs = self.softmax(self.prediction)
        return probs - self.target

    @staticmethod
    def softmax(x):
        logits = x - cp.max(x, axis=-1, keepdims=True)
        num = cp.exp(logits)
        den = cp.sum(num, axis=-1, keepdims=True)
        return num / den
    
    @staticmethod
    def logsoftmax(x):
        # LogSoftMax Implementation 
        max_x = cp.max(x, axis=-1, keepdims=True)
        exp_x = cp.exp(x - max_x)
        sum_exp_x = cp.sum(exp_x, axis=-1, keepdims=True)
        log_sum_exp_x = cp.log(sum_exp_x)
        max_plus_log_sum_exp_x = max_x + log_sum_exp_x
        log_probs = x - max_plus_log_sum_exp_x
        
        return log_probs
