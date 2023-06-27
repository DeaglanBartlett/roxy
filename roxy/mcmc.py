import numpyro.distributions as dist
from jax import lax

import roxy.likelihoods

class mylike_MNR(dist.Distribution):
   
    def __init__(self,):
        batch_shape = lax.broadcast_shapes(jnp.shape(A)) # FIX THIS
        super(mylike_MNR, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.marg_normal()  # ADD ARGS HERE
        
        
class mylike_prof(dist.Distribution):
   
    def __init__(self,):
        batch_shape = lax.broadcast_shapes(jnp.shape(A)) # FIX THIS
        super(mylike_MNR, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.marg_prof()  # ADD ARGS HERE
        

class mylike_unif(dist.Distribution):
   
    def __init__(self,):
        batch_shape = lax.broadcast_shapes(jnp.shape(A)) # FIX THIS
        super(mylike_MNR, self).__init__(batch_shape = batch_shape)
        
    def sample(self, key, sample_shape=()):
        raise NotImplementedError
        
    def log_prob(self, value):
        return - roxy.likelihoods.marg_unif()  # ADD ARGS HERE
