import jax
import jax.numpy as jnp
from flax import linen as nn

T = 1000
betas = jnp.linspace(0, 1, T)
alphas = 1 - betas


alphas_cumprod = jnp.cumprod(alphas)
alphas_cumprod_prev = jnp.append(1., alphas_cumprod[:-1])

sqrt_alphas_cumprod = jnp.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - alphas_cumprod)
log_one_minus_alphas_cumprod = jnp.log(1. - alphas_cumprod)
sqrt_recip_alphas_cumprod = jnp.sqrt(1. / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = jnp.sqrt(1. / alphas_cumprod - 1.)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_log_variance_clipped = jnp.log(jnp.append(posterior_variance[1], posterior_variance[1:]))
posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
posterior_mean_coef2 = (1. - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1. - alphas_cumprod)




def mse_loss(a, b):
    return jnp.mean((a-b)**2)

print(betas.shape, alphas.shape, alphas_cumprod.shape, alphas_cumprod_prev.shape)