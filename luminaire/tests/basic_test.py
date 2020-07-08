# write functional, low overhead tests like this
def test_has_tests():
    pass
    # assert_true (False, "There aren't any tests.")


def test_import_gitlab_egg_test():
    import luminaire  # noqa: F401 - suppress one type of error on this line
