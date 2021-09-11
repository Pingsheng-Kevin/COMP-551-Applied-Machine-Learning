import numpy as np

euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

