import os

with open('test.csv', 'r') as f:
    for line in f.readlines()[1:]:
        fn, c = line.split(',')
        c = int(c)
        c = f"{c:05}"
        os.makedirs(c, exist_ok=True)
        os.rename(fn, os.path.join(c, fn))
