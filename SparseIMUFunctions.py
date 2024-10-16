from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

def train(X_train, y_train, n_jobs=-1):
    
    '''
    train a Random Forest Classifier with the given training data
    return model
    '''
    model = RandomForestClassifier(n_estimators = 100, max_depth=30, \
    random_state=4, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    
    return model

def test(model, X_test):
    '''
    Return predictions
    '''
    y_pred = model.predict(X_test)
    
    return y_pred

