"""PFSFOG — Pooled Full-shape Surveys: Forecasts Of Galaxies.

Joint multi-tracer, multi-survey Fisher analysis for full-shape galaxy
power spectrum forecasts. The framework pools Fisher information across
the disjoint regions of N overlapping spectroscopic surveys' footprints,
with per-region multi-tracer Fishers combined in a common parameter
space (cosmology shared globally; nuisance parameters unique per
(tracer, redshift bin)).

The current implementation realizes the N=2 case for the
Subaru/PFS x DESI overlap; the architecture (`embed_fisher`,
`combine_zbins_heterogeneous`, `volume_partitioned_zbin_fisher`)
generalizes to N>=3 by enumerating the 2^N - 1 disjoint footprint
regions.
"""
