import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set()


def linreg_model(m=2, n=100, e=250):
    """Plotea modelo de regresion lineal con parametros de pendiente m, ruido e y size n"""

    # seed para la obtencion de random numbers
    np.random.seed(42)

    # Genero datos
    x = np.linspace(0, 1000, n)
    y = m * x + np.random.randint(e, size=n)

    X = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Modelado
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # df para analisis
    df = pd.DataFrame({"x": x, "y": y.reshape(n, ), "y_pred": y_pred.reshape(n, )})
    df['Residuos +'] = 0
    df['Residuos -'] = 0
    df.loc[(df['y'] <= df['y_pred']), 'Residuos +'] = 1
    df.loc[(df['y'] > df['y_pred']), 'Residuos -'] = 1

    rpos = sum(df['Residuos +'])
    rneg = sum(df['Residuos -'])

    data = [rpos, rneg]

    return x, y, y_pred, data, mse


def linreg_plotter(m=2, n=100, e=250):
    """plotea datos entregados por linreg_model"""

    # Cargo datos
    x, y, y_pred, data, mse = linreg_model(m, n, e)

    # Ploteado
    _ = plt.plot(x, y, color='orange')
    _ = plt.plot(x, y_pred, color='blue')

    _ = plt.xlabel("Valores de x")
    _ = plt.ylabel("Valores de y")
    _ = plt.legend(['Valores reales', 'Valores modelados'], loc='lower right')
    _ = plt.title('Modelo de Regresion lineal por Least Squares Method')
    _ = plt.annotate('mse = {:.2f}'.format(mse), xy=[10, 2000])
    plt.show()


def pie_plot(m=2, n=100, e=250):
    """Pie plot en base a los datos generados en linreg_plotter"""

    # Cargo datos
    x, y, y_pred, data, mse = linreg_model(m, n, e)

    # ploteo
    _ = plt.pie(data, autopct='%1.1f%%')
    _ = plt.legend(['Residuos +', 'Residuos -'], loc='lower right')
    _ = plt.title('Residuos generados por Least Squares Method')
    plt.show()


linreg_plotter()
pie_plot()
