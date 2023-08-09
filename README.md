# IRIS-Detection
"IRIS Type Classification with Forest Classifiers:Through machine learning, the project demonstrates how these classifiers analyze distinctive features of IRIS flowers to make precise predictions about their species. 
### Import Libraries
```
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
```
### Loading the Data
```
iris = datasets.load_iris()
iris
```

### Obtaining Input Features as well as output

```
iris.feature_names
iris.target_names
```
### Assigning the input and output variables
```
x = iris.data
y = iris.target
# obtaining the shape
x.shape
y.shape
```

### Building The Model
```
clf = RandomForestClassifier()
clf.fit(x,y)

clf.feature_importances_
```

### Creating The Prediction
```
x[0]
clf.predict(x[[0]])
clf.predict([[5.1, 3.5, 1.4, 0.2]]) # sample data point
clf.predict_proba(x[[0]]) # 100 percent accuracy
clf.fit(iris.data, iris.target_names[iris.target])
```

### 80 / 20 Split
```
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
```
To rebuild the model, we may perform the following
```
clf.fit(x_train, y_train)
clf.predict([[5.1, 3.5, 1.4, 0.2]])
clf.predict(x_test)
y_test
clf.score(x_test,y_test)
```





