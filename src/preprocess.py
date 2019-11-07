import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np

def main():
    cols = ['d_age', 'samerace', 'attractive_partner',
            'interests_correlate', 'like', 'guess_prob_liked', 'match']
    df = pd.read_csv('raw/data.csv', usecols=cols)
    
    df = df.replace('?', np.nan)
    df = df.dropna()
    
    df = df.sample(frac=1)
    nrows = df.shape[0]
    
    label = df['match']
    df = df.drop(['match'], axis=1)
    
    n = int(0.8 * nrows)
    trainX = df.iloc[: n]
    testX = df.iloc[n: ]
    
    trainY = label.iloc[: n]
    testY = label.iloc[n: ]

    rf = RFC(n_estimators=10)
    X = trainX.values
    y = trainY.values
    rf.fit(X, y)

    Xtest = testX.values
    Ytest = testY.values

    predicted = rf.predict(Xtest)
    out = Ytest == predicted
    out = np.where(out == True)

    acc = len(out[0]) / Ytest.shape[0]
    print(acc)

    # write match test
    pd.DataFrame(testY.index).to_csv('preprocessed/match_test.txt', header=False, sep='\t', index=False)

    # write match obs
    pd.DataFrame(trainY).to_csv('preprocessed/match_train.txt', header=False, sep='\t')

    # write rf predictions
    outdf = pd.DataFrame(predicted, index=testX.index, columns=['predicted'])
    outdf.to_csv('preprocessed/rf.txt', header=False, sep='\t')

    # write similarities
    out = []
    t = testX['interests_correlate']
    for i, r in t.items():
        for j, rr in t.items():
            rr = float(rr) / 2 + 0.5
            out.append(f'{i}\t{j}\t{rr}')
    s = '\n'.join(out)
    with open('preprocessed/sim.txt', 'w') as f:
        f.write(s)


if __name__ == '__main__':
    main()



















