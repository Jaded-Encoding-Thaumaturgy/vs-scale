from vskernels import Kernel, get_kernel

__all__ = [
    'CompareSameKernelError'
]


class CompareSameKernelError(ValueError):
    """Raised when two of the same kernels are compared to each other."""

    def __init__(
        self, function: str, kernel: Kernel | str,
        message: str = "{func}: 'You may not compare {kernel} with itself!'"
    ) -> None:
        self.function: str = function
        self.message: str = message

        if isinstance(kernel, str):
            kernel = get_kernel(kernel)()

        self.kernel: Kernel = kernel

        super().__init__(self.message.format(func=self.function, kernel=self.kernel.__class__.__name__))
