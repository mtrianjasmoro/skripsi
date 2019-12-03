#import data
import vega_datasets

import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from sklearn import svm

#ambil data
iris = vega_datasets.data.iris()
#hapus data spesisies virginica
iris = iris.drop(iris[iris['species'] == 'virginica'].index)

#menampilkan species
set(iris['species'])

#data spesies
iris.head()

#perbandingan data dari grafig
sn.pairplot(iris, hue='species')

#perhitungan svm
clf = svm.SVC(kernel='linear')
x = iris[['sepalWidth','petalWidth']]
y = iris['species']
clf.fit(x,y)

#membuat grafik svm
colors = {'setosa' : 'b', 'versicolor' : 'r'}
plt.scatter(iris['sepalWidth'],iris['petalWidth'],c=[colors[r] for r in iris['species']])
#set grafik svm
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0],xlim[1],30)
yy = np.linspace(ylim[0],ylim[1],30)
XX,YY = np.meshgrid(xx,yy)
xy = np.vstack([XX.ravel(),YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX,YY,Z, colors='k', levels = [-1,0,1], alpha=0.5, linestyles=['--','-','--'])

#prediksi svm
clf.predict([[3.0,0.5]])

