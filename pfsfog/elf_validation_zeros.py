"""Exact Q-multipole angular support for the retained ELF physical model."""
import numpy as np


def counterterm_zero_support(names, pairs, *, alphas, settings):
    """Declare orthogonality only for identity AP and an isotropic P_lin base.

    nuisance_to_backend maps Q0,Q2,Q4 onto L0,(2f/3)L2,(8f²/35)L4.
    The shared-field completion preserves these fitted auto multipoles only.
    Unequal-tracer cross multipoles need not be orthogonal. The
    geometry tolerance admits only float64 roundoff in equal distance ratios;
    it is not a derivative acceptance tolerance.
    """
    alphas = np.asarray(alphas, dtype=float)
    identity = (alphas.shape == (2,) and np.isfinite(alphas).all()
                and np.all(np.abs(alphas-1.) <= 32*np.finfo(float).eps))
    isotropic = (settings.get("engine") == "elf.EPT"
                 and settings.get("counterterm_base") == "linear"
                 and settings.get("do_irres") is False)
    if not identity or not isotropic:
        return ()
    support = []
    for name in names:
        parts = name.split(":")
        if len(parts) != 3 or parts[2] not in ("c0", "c2", "c4"):
            continue
        active_ell = int(parts[2][1:])
        for pair in pairs:
            if len(pair) == 2 and pair[0] == pair[1] == parts[1]:
                support += [(name, tuple(pair), ell) for ell in (0, 2, 4) if ell != active_ell]
    return tuple(support)
