import pandas as pd
from sklearn import metrics


def clust_eval(true, pred):
    x = metrics.adjusted_rand_score(true, pred)
    print(" Adjusted Rand index Score is : %.2f " % x)
    rand_score = x
    x = metrics.adjusted_mutual_info_score(true, pred)
    print(" Mutual Information based scores is : ", x)
    mutual_info_score = x
    x = metrics.homogeneity_score(true, pred)
    print(" Homogeneity scores is: ", x)
    homogeneity_score = x
    x = metrics.completeness_score(true, pred)
    print(" completeness scores is: ", x)
    completeness_score = x
    x = metrics.v_measure_score(true, pred)
    print(" V_Measure scores is: ", x)
    v_measure_score =  x
    x = metrics.fowlkes_mallows_score(true, pred)
    print(" Fowlkes-Mallows scores is: ", x)
    fowlkes_mallows_score = x
    d = {'Adjusted Rand index': [rand_score], 'Mutual Information based': [mutual_info_score], 'Homogeneity scores': [homogeneity_score],
         'completeness scores':[completeness_score], 'V_Measure scores':[v_measure_score],'Fowlkes-Mallows scores':[fowlkes_mallows_score]}
    df = pd.DataFrame(data=d)
    print(df)
    cluste_value = rand_score + mutual_info_score + homogeneity_score + completeness_score +v_measure_score + fowlkes_mallows_score

    return cluste_value
