import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)

in_state = np.load('in_state.npy')
in_label = np.zeros(shape=(in_state.shape[0])) + 1

out_state = np.load('out_state.npy')
out_label = np.zeros(shape=(out_state.shape[0])) + 2

not_state = np.load('not_state.npy')
not_state = not_state[np.random.choice(range(not_state.shape[0]), replace=False, size=2000)]
not_label = np.zeros(shape=not_state.shape[0])

print(in_state.shape, out_state.shape, not_state.shape)
X = np.concatenate([in_state, out_state, not_state])
y = np.concatenate([in_label, out_label, not_label])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, shuffle=True, test_size=0.2)

# Univariate feature selection with F-test for feature scoring
# We use the default selection function to select the four
# most significant features

X_indices = np.arange(X.shape[-1])
X_val = ["x", "y", "x'", "y'", "x''", "y''"]
selector = SelectKBest(f_classif, k=2)
selector.fit(X_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
plt.bar(X_val, scores, width=.2,
        label=r'Univariate score ($-Log(p_{value})$)')
plt.title("Univariate feature selection with F-test for feature scoring")
plt.legend()
plt.show()

clf = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=100))
clf.fit(X_train, y_train)
print('Classification accuracy: {:.3f}'.format(clf.score(X_test, y_test)))

plt.scatter(X_test[:, 4], X_test[:, 5], c=clf.predict(X_test), alpha=0.5)
plt.show()
