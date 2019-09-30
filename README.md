# python-utility
commonly used package


## printUniqueRatio
```python
import random
import utility

list_len = 10000
dest = ['剪刀', '石頭', '布', True, 1, 2, 3]

list_dest = [random.choice(dest) for _ in range(list_len)]
utility.printUniqueRatio(list_dest)

>>
index: 0 - 1: 28.73%(2873/10000)
index: 1 - 2: 13.64%(1364/10000)
index: 2 - 3: 14.67%(1467/10000)
index: 3 - 布: 14.58%(1458/10000)
index: 4 - 剪刀: 14.28%(1428/10000)
index: 5 - 石頭: 14.10%(1410/10000)
```
