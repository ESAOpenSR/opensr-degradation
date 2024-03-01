"""Tests for hello function."""

# UPDATEME by removing this file once the `hello` function is no longer needed for your
# project and `example.py` is also removed

import time

import pytest

from opensr_degradation.example import hello


def something(duration=0.000001):
    """Generic function to showcase benchmarking."""
    time.sleep(duration)

    return "Hello Aaron!"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Jeanette", "Hello Jeanette!"),
        ("Raven", "Hello Raven!"),
        ("Maxine", "Hello Maxine!"),
        ("Matteo", "Hello Matteo!"),
        ("Destinee", "Hello Destine!"),  # UPDATEME with the correct `expected` value
        ("Alden", "Hello Alden!"),
        ("Mariah", "Hello Mariah!"),
        ("Anika", "Hello Anika!"),
        ("Isabella", "Hello Isabella!"),
    ],
)
def test_hello(name, expected):
    """Example test with parametrization."""
    assert hello(name) == expected


def test_my_stuff(benchmark):
    """Example test with benchmark."""
    # Benchmark the something function with pytest-benchmark
    result = benchmark(something)

    # Extra code, to verify that the run completed correctly
    expected = hello("Aaron")

    # Sometimes you may want to check the result against your benchmark, writing faster
    # functions are no good if they return incorrect results
    assert result == expected
