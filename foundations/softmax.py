import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        s = 0
        m = max(z)
        for n in z:
            s += math.e ** (n - m)
        return [np.round((math.e ** (n - m)) / s, 4) for n in z]
