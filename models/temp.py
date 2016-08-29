import numpy as np

# map
x = [0, 1, 2, 3, 4, 5]
map(np.sin, x)
map(lambda x: x**2, x)
[a**2 for a in x]

y = [a + 1 for a in x]
map(lambda x, y: (x + y)**2, x, y)


# filter
filter(lambda x: x > 2, x)
xn = np.array(x)
filtered_xn = filter(lambda x: x > 2, xn)
type(filtered_xn)


# reduce
reduce(lambda a, b: a + b, x)
reduce(lambda a, b: np.sqrt(a**2 + b**2), x)
reduce(lambda a, b: np.sqrt(a**2 + b**2), [])
reduce(lambda a, b: np.sqrt(a**2 + b**2), [], 0)


# enumerate
e = enumerate(x, start=4)
e.next()