"""A complete original attempt and real validation exits are both required."""
from copy import deepcopy

import pytest

from scripts.run_pfs_bias_broad import GRIDS
from scripts.run_pfs_bias_stochastic_validation import (
    original_finished, validate_original_grid, validate_validation_execution,
)


def test_finished_failed_attempt_is_retained_but_incomplete_attempt_cannot_advance():
    assert original_finished({'status':'running','controllers':[]}) is False
    rows=[{'grid':g,'actual_exit':int(g=='angular64')} for g in GRIDS]
    assert original_finished({'status':'failed','controllers':rows}) is True
    for bad in (rows[:-1],[rows[0]]*5,[{**r,'actual_exit':137} for r in rows]):
        with pytest.raises(ValueError):original_finished({'status':'failed','controllers':bad})


def test_original_grid_requires_every_exact_population_and_real_worker_exits():
    expected=[{'key':str(i)} for i in range(35)]
    rows=[{'key':str(i),'status':'failed' if i==2 else 'passed','exit_code':int(i==2)} for i in range(35)]
    summary={'status':'failed','workers':rows}
    validate_original_grid(summary,expected)
    for workers in (rows[:-1],[rows[0]]*35,[{**r,'exit_code':137} for r in rows]):
        with pytest.raises(ValueError):validate_original_grid({**summary,'workers':workers},expected)
    with pytest.raises(ValueError):validate_original_grid({**summary,'status':'running'},expected)


def test_successful_validation_requires_five_distinct_actual_zero_exits():
    record={'status':'passed','controllers':[{'grid':g,'actual_exit':0} for g in GRIDS]}
    validate_validation_execution(record)
    for status in ('running','failed'):
        with pytest.raises(ValueError):validate_validation_execution({**record,'status':status})
    bad=deepcopy(record);bad['controllers'][0]['actual_exit']=1
    with pytest.raises(ValueError):validate_validation_execution(bad)
