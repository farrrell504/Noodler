def check_if_interactive():
    """
    checks if ipython is being used
    """
    import __main__ as main

    if not hasattr(main, "__file__"):
        return True
    return False


def check_if_notebook():
    """
    crude way of knowing if using notebook/jupyter or just ipython
    """
    import ipynbname

    if check_if_interactive():
        try:
            ipynbname.name()
            return True
        except FileNotFoundError:
            return False
    else:
        return False