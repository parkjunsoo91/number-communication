def int_to_binary(d):
    fs = '{0:0' + str(8) + 'b}'
    b_str = fs.format(d)
    return [int(c) for c in b_str]

class bah():
    def __init__(self):
        dic = {"hell": 2}
        
        for p in dic:
            print(p, dic[p])
            setattr(self, p, dic[p])
#print(int_to_binary(2))
ba = bah()
print(ba.hell)

