import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from matplotlib.colors import ListedColormap

dataset = pd.read_csv('social-network-ads.csv')
X = dataset.iloc[:, [2, 3]].values
Y = dataset.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(
                                    X, 
                                    Y, 
                                    test_size=0.25, 
                                    random_state=0
)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors = 2)
classifier.fit(X_train, Y_train)



Y_pred = classifier.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_test)):
    plt.scatter(X_test[Y_test == j, 0], X_test[Y_test == j, 1],
        c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Класифікатор (test set)')
plt.xlabel('Вік')
plt.ylabel('Орієнтовна Зарплата')
plt.legend()
plt.show()