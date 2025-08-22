#Entropy 

def entropy(y):
    p=y.value_counts(normalize=True).values
    ent = -np.sum(p*np.log2(p+1e-9)) # adding small term to avoid log0
    return ent

#Weighted Entropy

def calculate_weighted_entropy(feature,y):
    categories = feature.unique()

    weighted_entropy = 0

    for category in categories:
        y_category = y[feature == category]
        entropy_category = entropy(y_category)
        weighted_entropy += y_category.shape[0]/y.shape[0]*entropy_category


    return weighted_entropy

#Information Gain

def information_gain(feature,y):
    parent_entropy = entropy(y)

    child_entropy = calculate_weighted_entropy(feature,y)

    ig = parent_entropy - child_entropy

    return ig

#Gini Impurity

def gini_impurity(y):
    p=y.value_counts(normalize=True).values
    gini = 1-np.sum(p**2)
    
    return gini

def calculate_weighted_entropy_gini(feature,y):
    categories = feature.unique()

    weighted_gini_entropy = 0

    for category in categories:
        y_category = y[feature == category]
        gini_entropy_category = gini_impurity(y_category)
        weighted_gini_entropy += y_category.shape[0]/y.shape[0]*gini_entropy_category


    return weighted_gini_entropy

def information_gain(feature,y):
    parent_gini = gini_impurity(y)
    
    child_gini = calculate_weighted_entropy_gini(feature,y)
    
    ig= parent_gini- child_gini
    
    return ig