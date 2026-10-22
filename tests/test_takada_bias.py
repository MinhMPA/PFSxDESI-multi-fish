"""The old low-redshift bias experiment must include the newly populated cell."""
import pytest


def test_low_redshift_bias_window_follows_corrected_support_only():
    from scripts.run_takada_bias import corrected_bias_window
    assert corrected_bias_window([.7,.8],.6)==(.6,.8)
    assert corrected_bias_window([.7,.8],.7)==(.7,.8)
    assert corrected_bias_window([1.4,1.6],.6)==(1.4,1.6)
    assert corrected_bias_window([1.6,2.],.6)==(1.6,2.)
    with pytest.raises(ValueError):corrected_bias_window([.8,1.],.6)
