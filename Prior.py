
class Prior():
    '''
    prior_type:'uniform','gaussian'
    min,max: bound
    mu,sigma: for 'gaussian'
    '''
    def __init__(self,prior_type,min,max,mu = None,sigma = None):
        self.prior_type = prior_type
        self.min = min
        self.max = max
        self.mu = mu
        self.sigma = sigma

def priors_from_bound(bounds):
    '''
    input: 2D-array,bounds
    output:1D list for priors
    '''
    priors = []
    for bound in bounds:
        priors.append(Prior('uniform',bound[0],bound[1]))
    return priors