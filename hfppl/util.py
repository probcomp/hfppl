import numpy as np

def logsumexp(nums):
    m = np.max(nums)
    return np.log(np.sum(np.exp(nums - m))) + m
    
def log_softmax(nums):
    return nums - logsumexp(nums)

def softmax(nums):
    return np.exp(log_softmax(nums))