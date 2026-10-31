import pytest
from tccdet.prompts import build_prompts


def test_unique_and_deterministic():
    a = build_prompts("cat", 300, seed=1)
    assert len(set(a)) == 300 and a == build_prompts("cat", 300, seed=1)


def test_error_when_too_many():
    with pytest.raises(ValueError):
        build_prompts("cat", 10**6)
