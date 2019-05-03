def _to_str_hash(features):
    """
    Compresses features into a hash. E.g. [1,2,3] -> '123'.
    """
    return ''.join(map(lambda f: str(int(f)), features))


class HashFeatureTransformer:

    def __init__(self, env):
        """
        Compresses features into a hash. E.g. [1,2,3] -> '123'.
        """
        pass

    def transform(self, observations):
        return _to_str_hash(observations)


class TigerFeatureTransformer:

    def __init__(self, env):
        """
        Maps
            [1, 0, 0] -> 0
            [0, 1, 0] -> 1
            [0, 0, 1] -> 2
        """
        pass

    def transform(self, o):
        if o == [1, 0, 0]:
            return 0
        elif o == [0, 1, 0]:
            return 1
        elif o == [0, 0, 1]:
            return 2
        else:
            raise ValueError('Invalid observation: '.format(o))
