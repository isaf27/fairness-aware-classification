import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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
    
def smote_resampling(kdtree, X, y, N, k=11):
    res_x, res_y = [], []
    for n in range(N):
        i = np.random.randint(0, len(X) - 1)
        _, neigbs = kdtree.query(X[i:i + 1], k=k)
        neigb = neigbs[0, np.random.randint(0, k - 1)]
        a = np.random.rand(X[i].shape[0])
        
        res_x.append((1 - a) * X[i] + a * X[neigb])
        ys = y[neigbs[0]]
        res_y.append(max(set(ys), key=lambda g: (ys == g).sum()))
    
    return np.array(res_x), np.array(res_y)
    
class SMOTEBoost:
    def __init__(self, n_estimators=50, base_classifier=lambda: DecisionTreeClassifier(max_depth=1)):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
    
    def fit(self, X, y, is_protected):
        y = 2 * y - 1
        m = len(X)
        D = np.ones(m, dtype=float) / m
        self.classifiers = []
        self.alphas = []
        N = max(1, (1 - is_protected).sum() - is_protected.sum())
        
        tree = KDTree(X[is_protected], leaf_size=2)
        for i in range(self.n_estimators):
            smote_X, smote_y = smote_resampling(tree, X[is_protected], y[is_protected], N)
            new_X, new_y = np.vstack((X, smote_X)), np.append(y, smote_y)
            new_D = np.ones(N, dtype=float) / N
            D = np.append(D, new_D)
            D /= D.sum()
            
            self.classifiers.append(self.base_classifier())     
            self.classifiers[-1].fit(new_X, new_y, sample_weight=D)
            y_prob = self.classifiers[-1].predict_proba(new_X)
            
            likelihood = y_prob[:, 1]
            likelihood[new_y == -1] = 1 - likelihood[new_y == -1]
            eps = (D * likelihood).sum()
            beta = eps / (1 - eps)
            self.alphas.append(-np.log(beta))
            w = likelihood
            D = (D * np.power(beta, w))[:X.shape[0]]
            D /= D.sum()
    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        out = sum(alpha * classifier.predict(X)
                  for alpha, classifier in zip(self.alphas, self.classifiers)) > 0
        return out
