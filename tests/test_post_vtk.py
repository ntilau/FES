"""Tests for the VTK writers' text formatting.

``mesh::Savefield`` and ``field::field`` are deterministic text writers, so
the numeric formatting must match the reference output exactly.
"""

from __future__ import annotations

from fes.post.field import _f32, freq_tag
from fes.post.vtk import _c_int


def test_float_formatting_matches_the_reference_stream() -> None:
    # ``std::stringstream << double`` and ``std::ostream << (float)`` both use
    # the default six significant digits.
    assert freq_tag(1e10) == "1e+10"
    assert freq_tag(299792458.0) == "2.99792e+08"
    assert freq_tag(150e9) == "1.5e+11"
    assert _f32(1.0 / 3.0) == "0.333333"
    assert _f32(-0.0) == "-0"
    # ``(int)`` of an arma::uvec element wraps on the reference implementation's targets.
    assert _c_int(0xFFFFFFFF) == -1
    assert _c_int(3) == 3


