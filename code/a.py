import pandas as pd
import numpy as np

result = pd.DataFrame(np.arange(6).reshape(3,2), index=['a','b','c'], columns=['r', 's'])

print(result)

print(np.linspace(1, 10, 3))