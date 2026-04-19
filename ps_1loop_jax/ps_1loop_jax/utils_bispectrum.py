import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp

@jit
def _safe_sqrt(x):
    return jnp.sqrt(jnp.maximum(x, 0.0))

@jit
def get_muij(ki, kj, kk):
    """
    Compute the cosine angle between two wavevectors, muij=cosine(thetaij)=(ki*kj)/(|ki|*|kj|).

    Parameters
    ----------
    ki,kj,kk: array-like
        The Fourier wavenumbers k=|vk|, assuming the input triplets satistfy the closed triangle condition vki+vkj+vkk=0.

    Returns
    ----------
    muij: array-like
        The cosine angle muij=cosine(thetaij)=(vki*vkj)/(ki*kj).
    """
    muij = (kk**2 - ki**2 - kj**2) / (2*ki*kj)
    muij = jnp.clip(muij, -1.0, 1.0)
    return muij

@jit
def get_mu2_mu3(k1, k2, k3, mu1, phi):
    """
    Compute the LOS angles mu2 and mu3 from mu1 and phi.

    Parameters
    ----------
    k1,k2,k3: array-like
        The Fourier wavenumbers, assuming k1<=k2<=k3 and vk1+vk2+vk3=0.
    mu1: array-like
        The cosine angle of the first wavevector, mu1=cosine(theta)=(k1*z)/(|k1|*|z|) where zhat is the LOS direction.
    phi: array-like
        The azimuthal angle around k1, i.e. the angle between the projection of k2 onto the plane perpendicular to k1 and the x-axis.

    Returns
    ----------
    mu2, mu3: array-like
        The cosine angles mu2=cosine(theta2)=(k2*z)/(|k2|*|z|) and mu3=cosine(theta3)=(k3*z)/(|k3|*|z|).
    """
    mu12 = get_muij(k1, k2, k3)
    mu2 = mu1 * mu12 - _safe_sqrt(1 - mu1**2) * _safe_sqrt(1 - mu12**2) * jnp.cos(phi)
    mu3 = -k1 / k3 * mu1 - k2 / k3 * mu2
    mu2 = jnp.clip(mu2, -1.0, 1.0)
    mu3 = jnp.clip(mu3, -1.0, 1.0)
    return mu2, mu3

@jit
def get_triangle_vectors(k1, k2, k3, mu1, phi):
    """
    Build a closed triangle of wavevectors with the LOS chosen as the z-axis.

    Parameters
    ----------
    k1, k2, k3 : array-like
        Wavenumber magnitudes that satisfy the triangle inequality.
    mu1 : array-like
        Cosine between the first wavevector and the LOS.
    phi : array-like
        Azimuthal angle of the second wavevector around the first.

    Returns
    -------
    kvec1, kvec2, kvec3 : array-like
        Cartesian wavevectors of shape ``(..., 3)`` satisfying
        ``kvec1 + kvec2 + kvec3 = 0``.
    """
    mu12 = get_muij(k1, k2, k3)
    sin1 = _safe_sqrt(1 - mu1**2)
    sin12 = _safe_sqrt(1 - mu12**2)

    khat1 = jnp.stack((sin1, jnp.zeros_like(mu1), mu1), axis=-1)
    e_perp1 = jnp.stack((mu1, jnp.zeros_like(mu1), -sin1), axis=-1)
    e_perp2 = jnp.broadcast_to(jnp.array([0.0, 1.0, 0.0]), khat1.shape)

    khat2 = (
        mu12[..., None] * khat1
        + sin12[..., None]
        * (
            jnp.cos(phi)[..., None] * e_perp1
            + jnp.sin(phi)[..., None] * e_perp2
        )
    )

    kvec1 = k1[..., None] * khat1
    kvec2 = k2[..., None] * khat2
    kvec3 = -(kvec1 + kvec2)

    return kvec1, kvec2, kvec3

@jit
def get_ap_ref_triangle(k1, k2, k3, mu1, phi, alpha_perp, alpha_para):
    """
    Map an observed triangle to the true triangle under Alcock-Paczynski scaling.

    Returns
    -------
    tuple
        ``(k1_true, k2_true, k3_true, mu1_true, mu2_true, mu3_true,
        mu12_true, mu13_true, mu23_true)``.
    """
    kvec1_obs, kvec2_obs, kvec3_obs = get_triangle_vectors(k1, k2, k3, mu1, phi)

    scale = jnp.array([1.0 / alpha_perp, 1.0 / alpha_perp, 1.0 / alpha_para])
    kvec1 = kvec1_obs * scale
    kvec2 = kvec2_obs * scale
    kvec3 = kvec3_obs * scale

    k1_true = jnp.linalg.norm(kvec1, axis=-1)
    k2_true = jnp.linalg.norm(kvec2, axis=-1)
    k3_true = jnp.linalg.norm(kvec3, axis=-1)

    mu1_true = jnp.clip(kvec1[..., 2] / k1_true, -1.0, 1.0)
    mu2_true = jnp.clip(kvec2[..., 2] / k2_true, -1.0, 1.0)
    mu3_true = jnp.clip(kvec3[..., 2] / k3_true, -1.0, 1.0)

    mu12_true = jnp.clip(jnp.sum(kvec1 * kvec2, axis=-1) / (k1_true * k2_true), -1.0, 1.0)
    mu13_true = jnp.clip(jnp.sum(kvec1 * kvec3, axis=-1) / (k1_true * k3_true), -1.0, 1.0)
    mu23_true = jnp.clip(jnp.sum(kvec2 * kvec3, axis=-1) / (k2_true * k3_true), -1.0, 1.0)

    return (
        k1_true,
        k2_true,
        k3_true,
        mu1_true,
        mu2_true,
        mu3_true,
        mu12_true,
        mu13_true,
        mu23_true,
    )
