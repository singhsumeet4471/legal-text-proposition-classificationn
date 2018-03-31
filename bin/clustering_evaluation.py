from sklearn import metrics


def clust_eval(true, pred):
    x = metrics.adjusted_rand_score(true, pred)
    print(" Adjusted Rand index Score is : ", x)
    x = metrics.adjusted_mutual_info_score(true, pred)
    print(" Mutual Information based scores is : ", x)
    x = metrics.homogeneity_score(true, pred)
    print(" Homogeneity scores is: ", x)
    x = metrics.completeness_score(true, pred)
    print(" completeness scores is: ", x)
    x = metrics.v_measure_score(true, pred)
    print(" V_Measure scores is: ", x)
    x = metrics.fowlkes_mallows_score(true, pred)
    print(" Fowlkes-Mallows scores is: ", x)
