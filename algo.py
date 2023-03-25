import numpy as np
from tqdm.notebook import trange

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KDTree
from metrics import tpr_protected, tnr_protected, tnr_non_protected, tpr_non_protected, equalized_odds
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import RUSBoostClassifier

from catboost import CatBoostClassifier, Pool
from metrics import fnr_protected, fnr_non_protected, fpr_protected, fpr_non_protected


def _calculate_weights(y_true, y_pred, is_protected, C=0.7, eps=0.03):
    deltaFNR = fnr_protected(y_true, y_pred, is_protected) - fnr_non_protected(y_true, y_pred, is_protected)
    deltaFPR = fpr_protected(y_true, y_pred, is_protected) - fpr_non_protected(y_true, y_pred, is_protected)
    y_true = y_true.astype('bool')
    y_pred = y_pred.astype('bool')
    u = np.zeros(y_true.shape, dtype=float)
    s_plus = (np.logical_not(is_protected) & y_true)
    s_minus = (np.logical_not(is_protected) & np.logical_not(y_true))
    s_hat_plus = (is_protected & y_true)
    s_hat_minus = (is_protected & np.logical_not(y_true))
    U = 1.0 #1.0 - np.mean(y_true)
    if deltaFNR > eps:
        u += s_hat_plus.astype('float') * 1.0 * U
        u += s_plus.astype('float') * C * U
    elif deltaFNR < -eps:
        u += s_hat_plus.astype('float') * C * U
        u += s_plus.astype('float') * 1.0 * U
    else:
        u += s_hat_plus.astype('float') * 1.0 * U
        u += s_plus.astype('float') * 1.0 * U

    U = 1.0 #np.mean(y_true)
    if deltaFPR > eps:
        u += s_hat_minus.astype('float') * 1.0 * U
        u += s_minus.astype('float') * C * U
    elif deltaFPR < -eps:
        u += s_hat_minus.astype('float') * C * U
        u += s_minus.astype('float') * 1.0 * U
    else:
        u += s_hat_minus.astype('float') * 1.0 * U
        u += s_minus.astype('float') * 1.0 * U

    return u

class CatBoostReweight:
    def __init__(self, n_iter=10, cb_trees=10, C=0.7, eps=0.03, balance='SqrtBalanced'):
        self.n_iter = n_iter
        self.cb_trees = cb_trees
        self.C = C
        self.eps = eps
        self.balance = balance
    
    def fit(self, X, y, is_protected):
        self.models = []
        self.model = None
        res = np.zeros(X.shape[0])
        pool = Pool(X, label=y, baseline=res)
        for it in range(self.n_iter):
            if self.model is not None:
                pool.set_baseline(res)
                pool.set_weight(_calculate_weights(y, (res > 0), is_protected, C=self.C, eps=self.eps))
            self.model = CatBoostClassifier(
                iterations=self.cb_trees,
                verbose=0,
                random_seed=239,
                eta=0.1,
                thread_count=-1,
                auto_class_weights=self.balance
            )
            self.model.fit(pool)
            self.models.append(self.model)
            res += self.model.predict(X, prediction_type='RawFormulaVal')
            #print(f'{it} done!')
        return (res > 0)
    
    def predict(self, X):
        res = np.zeros(X.shape[0])
        for model in self.models:
            res += model.predict(X, prediction_type='RawFormulaVal')
        return (res > 0)

class AdaBoost:
    def __init__(self, n_estimators=50, base_classifier=lambda: DecisionTreeClassifier(max_depth=1)):
        self._adaboost = AdaBoostClassifier(base_classifier(), n_estimators=n_estimators)
    
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
    def __init__(self, n_estimators=100, base_classifier=lambda: DecisionTreeClassifier(max_depth=1), u_weight=5):
        self.n_estimators = n_estimators
        self.base_classifier = base_classifier
        self.u_weight = u_weight
    
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
            
            u = self.u_weight * _calculate_u(y, current_predictions > 0, is_protected)
            distribution = _ada_boost_distribution(
                y, y_pred_t, np.abs(current_predictions), distribution, self.alphas[-1], u)
    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        out = sum(alpha * classifier.predict(X)
                  for alpha, classifier in zip(self.alphas, self.classifiers)) > 0
        return out
    
def _calculate_u_correct(y_true, y_pred, y_pred_cum, is_protected, eps=1e-5):
    deltaFNR = tnr_protected(y_true, y_pred_cum, is_protected) - tnr_non_protected(y_true, y_pred_cum, is_protected)
    deltaFPR = tpr_protected(y_true, y_pred_cum, is_protected) - tpr_non_protected(y_true, y_pred_cum, is_protected)
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
    
def _ada_boost_distribution_correct(y, y_pred_t, confidence, distribution, alpha_t, u):
    distribution *= np.exp(alpha_t * (y != y_pred_t) * confidence) * (1 + u)
    distribution /= distribution.sum()
    return distribution

class AdaFairCorrect:
    def __init__(self, n_estimators=200, base_classifier=lambda: DecisionTreeClassifier(max_depth=2)):
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
            y_prob_t = self.classifiers[-1].predict_proba(X)[:, 1]
            
            alpha = _ada_boost_alpha(y, y_pred_t, distribution)
            self.alphas.append(alpha)
            current_predictions += alpha * y_pred_t
            
            u = _calculate_u_correct(y, y_pred_t, current_predictions > 0, is_protected)
            distribution = _ada_boost_distribution_correct(
                y, y_pred_t, np.abs(2 * y_prob_t - 1), distribution, self.alphas[-1], u)
    
    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])
        out = sum(alpha * classifier.predict(X)
                  for alpha, classifier in zip(self.alphas, self.classifiers)) > 0
        return out
    
def smote_resampling(kdtree, X, y, N, k=11):
    res_x, res_y = [], []
    for n in range(N):
        i = np.random.randint(0, len(X) - 1)
        dst, neigbs = kdtree.query(X[i:i + 1], k=k)
        neigb = neigbs[0, np.random.randint(0, k - 1)]
        a = np.random.rand(X[i].shape[0])
        new_sample = (1 - a) * X[i] + a * X[neigb]
        res_x.append(new_sample)
        y_neigbs = y[neigbs[0]]
        res_y.append(np.unique(y_neigbs, return_counts=True)[1].argmax())
    return np.array(res_x), np.array(res_y)

def _take_along_y(arr, y):
    res = np.zeros_like(arr[:, 0])
    res[y == 1] = arr[:, 1][y == 1]
    res[y == 0] = arr[:, 0][y == 0]
    return res

class SMOTEBoost:
    def __init__(self, n_estimators=10, base_classifier=lambda: LogisticRegression(), sample_protected=False):
        self.base_classifier = base_classifier  
        self.n_estimators = n_estimators
        self._sample_protected = sample_protected

    def fit(self, X, y, is_protected):
        m = len(X)
        D = np.ones_like(X) / len(y)
        for i in range(len(y)):
            D[i, y[i]] = 0
        self.classifiers = []
        self.alphas = []
        N = 100

        if self._sample_protected:
            X_protected = X[is_protected == 1]
            y_protected = y[is_protected == 1]
        else:
            protected_class = 1 if y.mean() < 0.5 else 0
            X_protected = X[y == protected_class]
            y_protected = np.full(X_protected.shape[0], fill_value=protected_class)
        
        tree = KDTree(X_protected, leaf_size=2)
        for i in range(self.n_estimators):
            smote_X, smote_y = smote_resampling(tree, X_protected, y_protected, N)
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

class SMOTEBoostProtected(SMOTEBoost):
    def __init__(self, n_estimators=10, base_classifier=lambda: LogisticRegression()):
        super(SMOTEBoostProtected, self).__init__(n_estimators, base_classifier, True)

class RUSBoost:
    def __init__(self):
        self._clf = RUSBoostClassifier()
    
    def fit(self, X, y, is_protected):
        self._clf.fit(X, y)
    
    def predict(self, X):
        return self._clf.predict(X)
