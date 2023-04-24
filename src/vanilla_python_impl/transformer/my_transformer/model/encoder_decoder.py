class EncoderDecoder:
    """Basic class of EncoderDecoder Model."""

    def __init__(self):
        pass

    def _build(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
