import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

def linreg_plotter(m=2, n=100, e=250):
    """Plotea modelo de regresion lineal con parametros de pendiente m, ruido e y size n"""

    # Genero datos
    x = np.linspace(0, 1000, n)
    y = m * x + np.random.randint(e, size=n)

    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Modelado
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Ploteado
    _ = plt.plot(X, y_pred)
    _ = plt.plot(X, y)
    plt.show()


linreg_plotter()