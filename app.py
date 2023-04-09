from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

import streamlit as st

st.title("Iris Classification")
st.sidebar.title("Input Parameters")

sepal_length = st.sidebar.slider("Sepal length", 0.0, 10.0, 5.0)
sepal_width = st.sidebar.slider("Sepal width", 0.0, 10.0, 5.0)
petal_length = st.sidebar.slider("Petal length", 0.0, 10.0, 5.0)
petal_width = st.sidebar.slider("Petal width", 0.0, 10.0, 5.0)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.sidebar.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Predicted iris species: {iris.target_names[prediction[0]]}")
