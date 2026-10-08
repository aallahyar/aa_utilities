import numpy as np
import pytest

from aa_utilities.graphics import adjust_brightness


@pytest.mark.parametrize(
    'color, rate, expected',
    [
        ('red', 0.5, '#800000'),
        ('red', 1.5, '#ff8080'),
        ('#808080', 2, '#ffffff'),  # lightness is clipped at 1
        ('#0000ff', 0, '#000000'),
        ('#3a7bd5', 1.0, '#3a7bd5'),
    ],
)
def test_adjusts_lightness_and_returns_hex(color, rate, expected):
    assert adjust_brightness(color, rate) == expected


@pytest.mark.parametrize(
    'color',
    ['red', 'tab:red', (1.0, 0.0, 0.0), [1.0, 0.0, 0.0], np.array([1.0, 0.0, 0.0])],
    ids=['name', 'tab-name', 'tuple', 'list', 'ndarray'],
)
def test_accepts_any_matplotlib_color(color):
    assert adjust_brightness(color, 1.0).startswith('#')
    assert len(adjust_brightness(color, 1.0)) == 7


def test_alpha_is_dropped_by_default():
    assert adjust_brightness((1, 0, 0, 0.5), 0.5) == '#800000'


def test_include_alpha_keeps_the_input_alpha():
    assert adjust_brightness((1, 0, 0, 0.5), 0.5, include_alpha=True) == '#80000080'


def test_include_alpha_on_opaque_color_appends_ff():
    assert adjust_brightness('red', 0.5, include_alpha=True) == '#800000ff'
