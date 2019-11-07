import pickle
import pandas as pd
import numpy as np
def main():
    a = []
    Ytest = []
    df = pd.read_csv('raw/data.csv', usecols=['match'])
    with open('inferred-predicates/MATCH.txt', 'r') as f:
        for line in f:
            l = line.split()
            Ytest.append(df['match'][int(l[0][1:-1])])
            a.append(round(float(l[1])))
    
    a = np.array(a)
    Ytest = np.array(Ytest)
    out = Ytest == a
    out = np.where(out == True)

    acc = len(out[0]) / Ytest.shape[0]
    print(acc)

if __name__ == '__main__':
    main()
