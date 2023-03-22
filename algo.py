import numpy as np
from tqdm.notebook import trange

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KDTree
from metrics import tpr_protected, tnr_protected, tnr_non_protected, tpr_non_protected

class AdaBoost:
    def __init__(self):
        self._adaboost = AdaBoostClassifier()
    
    def fit(self, X, y, is_protected):
        self._adaboost.fit(X, y)
    
    def predict(self, X):
        return self._adaboost.predict(X)

def _calculate_u(y_true, y_pred, is_protected, eps=1e-5):
    deltaFNR = tnr_protected(y_true, y_pred, is_protected) - tnr_non_protected(y_true, y_pred, is_protected)
    deltaFPR = tpr_protected(y_true, y_pred, is_protected) - tpr_non_protected(y_true, y_pred, is_protected)
    u = np.zeros(y_true.shape, dtype=float)
    s_plus = ((~is_protected) & y_true)
    s_minus = ((~is_protected) & (~y_true))
    s_hat_plus = (is_protected & y_true)
    s_hat_minus = (is_protected & (~y_true))
    fnr_mask = (y_true != y_pred) & (((deltaFNR > eps) & s_plus) | ((deltaFNR < -eps) & s_hat_plus))
    fpr_mask = (y_true != y_pred) & (((deltaFPR > eps) & s_minus) | ((deltaFPR < -eps) & s_hat_minus))
    u[fnr_mask] = np.abs(deltaFNR)
    u[fpr_mask] = np.abs(deltaFPR)
    return u

def _ada_boost_alpha(y, y_pred_t, distribution):
    eps = (distribution * (y != y_pred_t)).sum()
    return 0.5 * np.log((1.0 - eps) / eps)

def _ada_boost_distribution(y, y_pred_t, confidence, distribution, alpha_t, u):
    distribution *= np.exp(-alpha_t * y * y_pred_t * confidence) * (1 + u)
    distribution /= distribution.sum()
    return distribution

class AdaFair:
    def __init__(self, n_estimators=50, base_classifier=lambda: DecisionTreeClassifier(max_depth=1)):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
    
    def fit(self, X, y, is_protected):
        y = 2 * y - 1
        n_samples = len(X)
        distribution = np.ones(n_samples, dtype=float) / n_samples
        current_predictions = np.zeros(n_samples, dtype=float)
        self.classifiers = []
        self.alphas = []
        for i in range(self.n_estimators):
            self.classifiers.append(self.base_classifier())     
            self.classifiers[-1].fit(X, y, sample_weight=distribution)
            y_pred_t = self.classifiers[-1].predict(X)
            
            alpha = _ada_boost_alpha(y, y_pred_t, distribution)
            self.alphas.append(alpha)
            current_predictions += alpha * y_pred_t
            
            u = _calculate_u(y, current_predictions > 0, is_protected)
            distribution = _ada_boost_distribution(
                y, y_pred_t, np.abs(current_predictions), distribution, self.alphas[-1], u)
    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        out = sum(alpha * classifier.predict(X)
                  for alpha, classifier in zip(self.alphas, self.classifiers)) > 0
        return out
    
def smote_resampling(kdtree, X, N, k=11):
    res_x, res_y = [], []
    for n in range(N):
        i = np.random.randint(0, len(X) - 1)
        dst, neigbs = kdtree.query(X[i:i + 1], k=k)
        neigb = neigbs[0, np.random.randint(0, k - 1)]
        a = np.random.rand(X[i].shape[0])
        new_sample = (1 - a) * X[i] + a * X[neigb]
        res_x.append(new_sample)
    return np.array(res_x)

def _take_along_y(arr, y):
    res = np.zeros_like(arr[:, 0])
    res[y == 1] = arr[:, 1][y == 1]
    res[y == 0] = arr[:, 0][y == 0]
    return res

class SMOTEBoost:
    def __init__(self, n_estimators=10, base_classifier=lambda: DecisionTreeClassifier(max_depth=1)):
        self.base_classifier = base_classifier  
        self.n_estimators = n_estimators

    def fit(self, X, y, is_protected):
        m = len(X)
        D = np.ones_like(X) / len(y)
        for i in range(len(y)):
            D[i, y[i]] = 0
        self.classifiers = []
        self.alphas = []
        N = 10

        self.classes_ = np.array([0, 1])
        protected_class = 1 if y.mean() < 0.5 else 0
        X_protected = X[y == protected_class]
        
        tree = KDTree(X_protected, leaf_size=2)
        for i in range(self.n_estimators):
            smote_X = smote_resampling(tree, X_protected, N)
            smote_y = np.full(N, fill_value=protected_class)
            new_X, new_y = np.vstack((X, smote_X)), np.concatenate((y, smote_y))

            self.classifiers.append(self.base_classifier())
            self.classifiers[-1].fit(new_X, new_y)
            y_prob = self.classifiers[i].predict_proba(X)
            
            likelihood = _take_along_y(y_prob, y)
            dst_not_y = _take_along_y(D, 1 - y)
            eps = (2 * dst_not_y * (1 - likelihood)).sum()
            beta = eps / (1 - eps)
            self.alphas.append(-np.log(beta))
            D = (D * np.power(beta, 1 - likelihood)[:, None])
            D /= D.sum()
    
    def predict(self, X):
        out = sum(alpha * classifier.predict_proba(X)
                  for alpha, classifier in zip(self.alphas, self.classifiers)).argmax(1)
        return out