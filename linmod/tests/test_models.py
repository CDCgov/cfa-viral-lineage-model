from linmod import add_one, add_two


def test_add_one():
    assert add_one(1) == 2


def test_add_two():
    assert add_two(1) == 3
