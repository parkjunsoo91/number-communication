import os

seq = [4,5,6,8,10]
vocab = [2,3,4,5,6,7,8]
joint = ["", "--joint"]
for s in seq:
    for v in vocab:
        for j in joint:
            os.system("python sparse_game.py --cuda --max 100 --seq {} --vocab {} {} --epoch 50001".format(s, v, j))
