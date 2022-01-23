from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


data = pd.read_csv('./car-prices.csv')
change = {
    'yes': 1,
    'no': 0
}
data.sold = data.sold.map(change)

current_year = datetime.today().year
data['model_age'] = current_year - data.model_year
data['km_per_year'] = data.mileage_per_year * 1.60934
data = data.drop(columns=['Unnamed: 0', 'mileage_per_year', 'model_year'], axis=1)

x = data[['price', 'model_age', 'km_per_year']]
y = data['sold']

# Dummy
SEED = 20
np.random.seed(SEED)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)

dummy = DummyClassifier()
dummy.fit(train_x, train_y)
accuracy_rate = dummy.score(test_x, test_y)
print(f'Dummy accuracy rate: {accuracy_rate * 100}%')

# Decision Tree Classifier
SEED = 5
np.random.seed(SEED)
raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, test_size=0.25, stratify=y)

print(f'Trained with: {len(raw_train_x)} elements and tested with: {len(raw_test_x)}')

modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(raw_train_x, train_y)
predict = modelo.predict(raw_test_x)

accuary_rate = accuracy_score(test_y, predict)
print(f'Accuracy rate: {accuracy_rate * 100}%')

# graphviz
features = x.columns
dot_data = export_graphviz(modelo, out_file=None, filled=True, rounded=True,
                           feature_names=features, class_names=['no', 'yes'])

graphic = graphviz.Source(dot_data)
graphic.view()
