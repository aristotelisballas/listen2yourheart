class PureAbstractError(Exception):
    """
    Custom exception to be raised by pure abstract methods that should not be called.
    """

    def __init__(self):
        super().__init__("Do not call this method, it is a pure abstract method")
