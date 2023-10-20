import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('iris')
df.head()

df['species'], categories = pd.factorize(df['species'])
df.head()

df.describe

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(df.petal_length, df.petal_width, df.species)
ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('species')
plt.title('3D Scatter Plot Example')
plt.show()

sns.scatterplot(data=df, x="sepal_length",y="sepal_width",hue="species");

k_rng = range(1, 10)
sse=[]

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal_length', 'petal_width']])
    sse.append(km.inertia_)
    
plt.xlabel('k_rng')
plt.ylabel("sum of Squared errors")
plt.plot(k_rng, sse)   

k_rng = range(1, 10)
sse=[]

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['petal_length', 'petal_width']])
    sse.append(km.inertia_)
    
sse

plt.xlabel('k_rng')
plt.ylabel("sum of Squared errors")
plt.plot(k_rng, sse)

km = KMeans(n_clusters=3,random_state=0,)
y_predicted = km.fit_predict(df[['petal_length', 'petal_width']])
y_predicted

df['cluster']=y_predicted
df.head(150)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df.species, df.cluster)
cm

predicted_labels=df.cluster

cm = confusion_matrix(true_labels, predicted_labels)
class_labels = ['Setosa', 'versicolor', 'virginica']

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

for i in range (len(class_labels)):
    for j in range (len(class_labels)):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='red')
        
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()