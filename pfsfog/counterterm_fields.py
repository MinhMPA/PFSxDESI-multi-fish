"""CT-A deterministic counterterms from the same renormalized tracer fields.

The inputs retain the historical raw angular dictionary: ``c0,c2,c4``
multiply 1,f mu²,f² mu⁴ in -2 k² P_lin. Their units are length squared;
``cfog`` has units length to the fourth power. Priors remain in the caller's
Q=A c and T=A² tilde_c coordinates, before conversion to this dictionary.
"""
import jax.numpy as jnp


def leading_field_coefficients(ctr, b1, f):
    """Return e=(d0,d2/f,d4/f²) from raw ``ctr`` and current b1,f.

    The regular N system defines the continuous f=0 extension without growth
    divisions. It matches intrinsic auto multipoles 0,2,4 and retains the
    induced multipole 6. b1>0,f>=0 is the physical domain; no singular inverse
    is regularized. All amplitude, bias and growth dependence is differentiated.
    """
    c0 = ctr['c0'] + f*ctr['c2']/3 + f**2*ctr['c4']/5
    c2 = ctr['c2'] + 6*f*ctr['c4']/7
    c4 = ctr['c4']
    n = jnp.asarray([
        [b1+f/3, f*(b1/3+f/5), f**2*(b1/5+f/7)],
        [2/3, 2*b1/3+4*f/7, f*(4*b1/7+10*f/21)],
        [0., 8/35, 8*b1/35+24*f/77],
    ])
    return jnp.linalg.solve(n, -jnp.asarray([c0, 2*c2/3, 8*c4/35]))


def pair_counterterms(ctr_a, ctr_b, b_a, b_b, f):
    """Return ELF's (C0,C2,C4,C6,C44,C46,C48) for a tracer pair.

    Leading terms contract Z_a D_b+Z_b D_a. The single-coefficient FoG
    contraction is unchanged; an auto uses the same population for both sides.
    Neither growth nor either bias is held fixed inside the conversion.
    """
    a = leading_field_coefficients(ctr_a, b_a, f)
    b = leading_field_coefficients(ctr_b, b_b, f)
    fog = (ctr_a['cfog']+ctr_b['cfog'])/2
    return jnp.asarray([
        -(b_a*b[0]+b_b*a[0])/2,
        -(b_a*b[1]+b_b*a[1]+a[0]+b[0])/2,
        -(b_a*b[2]+b_b*a[2]+a[1]+b[1])/2,
        -(a[2]+b[2])/2,
        fog*b_a*b_b/2, fog*(b_a+b_b)/2, fog/2,
    ])
