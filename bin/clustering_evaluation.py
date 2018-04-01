from sklearn import metrics


def clust_eval(true, pred):
    x = metrics.adjusted_rand_score(true, pred)
    print(" Adjusted Rand index Score is : %.2f " % x)
    rand_score = 'Adjusted Rand index Score : %.2f ' % x
    x = metrics.adjusted_mutual_info_score(true, pred)
    print(" Mutual Information based scores is : ", x)
    mutual_info_score = '\n Mutual Information based scores is :%.2f ' % x
    x = metrics.homogeneity_score(true, pred)
    print(" Homogeneity scores is: ", x)
    homogeneity_score = '\n Homogeneity scores is: %.2f ' % x
    x = metrics.completeness_score(true, pred)
    print(" completeness scores is: ", x)
    completeness_score = '\n completeness scores is: %.2f ' % x
    x = metrics.v_measure_score(true, pred)
    print(" V_Measure scores is: ", x)
    v_measure_score = '\n V_Measure scores is: %.2f ' % x
    x = metrics.fowlkes_mallows_score(true, pred)
    print(" Fowlkes-Mallows scores is: ", x)
    fowlkes_mallows_score = '\n Fowlkes-Mallows scores is: %.2f ' % x
    cluste_value = rand_score + mutual_info_score + homogeneity_score + completeness_score +v_measure_score + fowlkes_mallows_score

    return cluste_value
