
def wave(n = 1):
    #xs = np.linspace(0, 50, 200)
    xs = range(1000)
    wavs = []
    mods = []
    for L in range (100,101):
        for x in xs:
            mod = smooth_mod(x+0.5, L, n)
            print("{:.1f} mod {} = {:.1f} ({:.1f})".format(x, L, x%L, mod))
            #Ls.append(L)
            wavs.append(mod)
            mods.append(x%L)
    
    plt.plot(xs, wavs)
    plt.plot(xs, mods)
    plt.show()

wave(20)