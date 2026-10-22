"""Explicit shell quadrature, independent of AP resolution and inverse arithmetic."""
from __future__ import annotations

LEGACY_SHELL_PROFILE = 'shell0025-v1'
FINE_SHELL_PROFILE = 'shell00125-v1'
SHELL_PROFILES = (LEGACY_SHELL_PROFILE, FINE_SHELL_PROFILE)


def shell_profile_record(name: str) -> dict:
    if name not in SHELL_PROFILES:
        raise ValueError('unknown shell profile')
    dk, count = (.0025, 76) if name == LEGACY_SHELL_PROFILE else (.00125, 152)
    return {'name': name, 'kmin': .01, 'kmax': .20, 'adopted_dk': dk,
            'control_dk': dk/2, 'adopted_shells': count, 'control_shells': 2*count,
            'original_grid_dk': .0025, 'quadrature': 'bounded-midpoint',
            'mode_weight': 'k^2*delta_k once in covariance'}


def shell_profile_from_record(record: dict, requested: str | None = None) -> str:
    if 'shell_profile' not in record:
        name = LEGACY_SHELL_PROFILE
    else:
        value = record['shell_profile']
        if not isinstance(value, dict) or value != shell_profile_record(value.get('name')):
            raise ValueError('shell profile declaration differs')
        name = value['name']
    if requested is not None and name != requested:
        raise ValueError('shell profile differs from requested study')
    return name


def shell_profile_fields(name: str) -> dict:
    value = shell_profile_record(name)
    return {} if name == LEGACY_SHELL_PROFILE else {'shell_profile': value}


def validate_shell_selection(name: str, ap_profile: str, solver: str) -> None:
    from .ap_numerics import INTRINSIC_PROFILE
    from .fisher_precision import DECIMAL_FISHER_SOLVER
    shell_profile_record(name)
    if name == FINE_SHELL_PROFILE and (ap_profile != INTRINSIC_PROFILE or solver != DECIMAL_FISHER_SOLVER):
        raise ValueError('candidate shell profile requires intrinsic AP16384 and Decimal50')


def validate_shell_configuration(config, profile: str, role: str) -> None:
    record = shell_profile_record(profile)
    spacing = {'adopted': record['adopted_dk'], 'shell_control': record['control_dk'],
               'original_grid': record['original_grid_dk']}
    if (role not in spacing or config.dk != spacing[role]
            or config.kmin != record['kmin'] or config.kmax != record['kmax']):
        raise ValueError('shell configuration or reference role differs')


def shell_fields(provenance: dict, config) -> dict:
    """Attach the cohort choice and actual role, without inferring from dk alone.

    A legacy shell control also has dk=.00125. Only an explicit producer
    declaration distinguishes that control from a candidate adopted grid.
    Legacy callers retain their historical, sometimes diagnostic, supports.
    """
    profile = shell_profile_from_record(provenance)
    if profile == LEGACY_SHELL_PROFILE:
        return {}
    declaration = shell_profile_record(profile)
    roles = {'adopted': declaration['adopted_dk'], 'shell_control': declaration['control_dk'],
             'original_grid': declaration['original_grid_dk']}
    role = next((key for key, dk in roles.items() if dk == config.dk), None)
    validate_shell_configuration(config, profile, role)
    return {'shell_profile': declaration, 'shell_grid_role': role}


def validate_shell_fields(record: dict, provenance: dict, config) -> None:
    expected = shell_fields(provenance, config)
    actual = {key: record[key] for key in ('shell_profile', 'shell_grid_role') if key in record}
    if actual != expected:
        raise ValueError('saved shell profile/role differs from producer or actual grid')
