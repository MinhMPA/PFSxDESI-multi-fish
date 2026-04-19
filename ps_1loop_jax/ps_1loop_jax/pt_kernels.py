import jax
jax.config.update('jax_enable_x64', True)
from jax import jit

@jit
def get_kijsum(ki,kj,muij):
    """
    Compute the magnitude of the vector sum vkij=vki+vkj.

    Parameters
    ----------
    ki, kj: array-like
        The Fourier wavenumbers.
    muij: array-like
        The cosine angle muij=cosine(thetaij)=(vki*vkj)/(ki*kj).

    Returns
    -------
    kijsum: array-like
        The magnitude kij=|vkij|=|vki+vkj|.
    """
    kijsum = (ki**2 + kj**2 + 2*ki*kj*muij)**0.5
    return kijsum

@jit
def get_mu_kijsum(ki, kj, mui, muj, kijsum):
    """
    Compute the cosine angle between vkij=vki+vkj and the line of sight.

    Parameters
    ----------
    ki, kj: array-like
        The Fourier wavenumbers.
    kijsum: array-like
        The magnitude of the vector sum vkij=vki+vkj.

    Returns
    -------
    mu_kk: array-like
        The cosine angle between vkk and the line of sight.
    """
    mu_kk = (ki * mui + kj * muj) / kijsum
    return mu_kk

@jit
def F2(ki,kj,muij):
    """
    Compute the symmetrized second-order SPT kernel for density, F2.

    Parameters
    ----------
    ki, kj: array-like
        The Fourier wavenumbers.
    muij: array-like
        The cosine angle between the two wavevectors, muij=cosine(thetaij)=(ki*kj)/(|ki|*|kj|).

    Returns
    -------
    F2: array-like
        The F2 kernel value.
    """
    F2 = 5/7 + 2/7*muij**2 + 1/2*muij*(ki/kj + kj/ki)
    return F2

@jit
def G2(ki,kj,muij):
    """
    Compute the symmetrized second-order SPT kernel for velocity divergence, G2.

    Parameters
    ----------
    ki, kj: array-like
        The Fourier wavenumbers.
    muij: array-like
        The cosine angle between the first and second wavevector, muij=cosine(thetaij)=(ki*kj)/(|ki|*|kj|).

    Returns
    -------
    G2: array-like
        The G2 kernel value.
    """
    G2 = 3/7 + 4/7*muij**2 + 1/2*muij*(ki/kj + kj/ki)
    return G2

@jit
def Z1(bias,f,mu):
    """
    Compute the redshift-space Z1 kernel.

    Parameters
    ----------
    bias: float
        Dictionary containing the linear bias parameters.
    f: float
        The linear growth rate.
    mu: array-like
        The cosine angle between the wavevector and the line of sight.

    Returns
    -------
    Z1: array-like
        The Z1 redshift-space kernel.
    """
    Z1 = bias['b1'] + f*mu**2
    return Z1

@jit
def Z1fog_ctr(bias,mu,k,kfog_nl=0.3):
    """
    Compute the FoG LO correction to the redshift-space Z1 kernel.

    Parameters
    ----------
    bias: float
        Dictionary containing the LO counterterm bias parameter.
    mu: array-like
        The cosine angle between the wavevector and the line of sight.
    k: array-like
        The Fourier wavenumbers.
    kfog_nl: float
        The characteristic Fourier wavenumber for FoG corrections. Default is 0.3.
    Returns
    -------
    Z1fog_ctr: array-like
        The FoG LO counterterm in the Z1fog kernel.
    """
    Z1fog_ctr = -bias['c1'] * mu**2 * (k/kfog_nl)**2
    return Z1fog_ctr

@jit
def Z1fog(bias,f,mu,k,kfog_nl=0.3):
    """
    Compute the redshif-space Z1fog kernel, including the LO counterterm to account for FoG effect.

    Parameters
    ----------
    bias: float
        Dictionary containing the LO counterterm bias parameter.
    f: float
        The linear growth rate.
    mu: array-like
        The cosine angle between the wavevector and the line of sight.
    k: array-like
        The Fourier wavenumbers.
    kfog_nl: float
        The characteristic Fourier wavenumber for FoG corrections. Default is 0.3.
    Returns
    -------
    Z1fog: array-like
        The Z1 redshift-space kernel including the FoG LO counterterm.
    """
    Z1fog = Z1(bias,f,mu) + Z1fog_ctr(bias,mu,k,kfog_nl=kfog_nl)
    return Z1fog

@jit
def Z2(ki,kj,bias,f,mui,muj,muij):
    """
    Compute the EFT Z2 kernel.

    Parameters
    ----------
    ki, kj: array-like
        The Fourier wavenumbers.
    bias: dict
        A dictionary containing bias parameters.
    f: float
        The linear growth rate.
    mui, muj: array-like
        The cosine angles between vki, vkj and the line of sight.
    muij: array-like
        The cosine angle muij=cosine(thetaij)=(vki*vkj)/(ki*kj).

    Returns
    -------
    Z2: array-like
        The Z2 kernel value.
    """
    kijsum = get_kijsum(ki, kj, muij)
    mu_kijsum = get_mu_kijsum(ki, kj, mui, muj, kijsum)
    Z2 = (
        0.5*bias['b2']
        + bias['bG2']*(muij**2-1)
        + bias['b1']*F2(ki,kj,muij)
        + f*mu_kijsum**2*G2(ki,kj,muij)
        + 0.5*f*mu_kijsum*kijsum*(
            mui/ki*(bias['b1']+f*muj**2)
            + muj/kj*(bias['b1']+f*mui**2)
        )
    )

    return Z2
