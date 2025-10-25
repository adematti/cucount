"""
Top-level cucount Python API.

This exposes a concise helper to compute 2-point correlations with optional
jackknife error estimates on top of the existing low-level count2 interface.

Example
-------
	import cucount
	# tracer_one, tracer_two are cucount.numpy.Particles
	corr, err = cucount.calculate_correlation(
		tracer_one, tracer_two, spin_tracer_one=0, spin_tracer_two=2,
		errors='jackknife', n_jackknife_regions=16,
	)

"""

from .correlations import calculate_correlation

__all__ = [
	"calculate_correlation",
]

