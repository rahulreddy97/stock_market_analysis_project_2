import datetime
import glob
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def set_matplot():
    # Adjusting the size of matplotlib
    mpl.rc('figure', figsize=(12, 8))
    mpl.__version__

    # Adjusting the style of matplotlib 
    style.use('ggplot') 


def correlation_analysis():
    set_matplot()
    path = str(Path().absolute()) + "/static"
    all_files = glob.glob(path + "/*.csv")

    li = []
    for filename in all_files:
        company = os.path.basename(filename).split(".")[0]
        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df[['Date', 'Adj Close']].rename(columns={'Adj Close': company})
                  .sort_values(by='Date')
                  .set_index('Date'))
    dfcomp = pd.concat(li, axis=1, join='inner')
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()

    # Correlation Analysis - Between different company
    # plt.scatter(retscomp.AAPL, retscomp.GOOG)
    # plt.xlabel('Return AAPL')
    # plt.ylabel('Return GOOG')

    # Correlation Analysis - using scatter_matrix map
    # scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10))

    # Correlation Analysis - using heat map
    # Notice that the lighter the color, the more correlated the two stock are.
    # plt.imshow(corr, cmap='hot', interpolation='none')
    # plt.colorbar()
    # plt.xticks(range(len(corr)), corr.columns)
    # plt.yticks(range(len(corr)), corr.columns)

    # Stocks Returns Rate and Risk
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(
            label,
            xy=(x, y), xytext=(20, -20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round, pad=0.5', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )
    # plt.show()


def predict_stock_price(company, start, end):
    set_matplot()
    # start = datetime.datetime(2010, 1, 1)
    # end = datetime.datetime(2019, 9, 11)

    df = web.DataReader(company, 'yahoo', start, end)

    # Feature Engineering
    dfreg = df[['Adj Close', 'Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Pre-processing and cross validation
    # Drop missing value na
    dfreg.fillna(value=-99999, inplace=True)

    # we want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.05 * len(dfreg)))

    # Separating the label here, we want to predict the Adj Close
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for liner regression
    X = preprocessing.scale(X)

    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Model Generation - Where the prediction fun stats
    # Linear regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)

    # Quadratic regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    # Evaluation
    confidence_reg = clfreg.score(X_test, y_test)
    confidencePoly2 = clfpoly2.score(X_test, y_test)
    confidencePoly3 = clfpoly3.score(X_test, y_test)
    confidenceKnn = clfknn.score(X_test, y_test)

    last_date = dfreg.iloc[-1].name

    dfreg['Linear'] = np.nan
    dfreg['Poly2'] = np.nan
    dfreg['Poly3'] = np.nan
    dfreg['Knn'] = np.nan

    lineardf = clfreg.predict(X_lately)
    poly2df = clfpoly2.predict(X_lately)
    poly3df = clfpoly3.predict(X_lately)
    fdf = clfknn.predict(X_lately)

    # Plotting the Prediction
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i, _ in enumerate(lineardf):
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns) - 4)] + [lineardf[i]] + [poly2df[i]] + [
            poly3df[i]] + [fdf[i]]

    # dfreg['Adj Close'].tail(500).plot()
    # dfreg['Linear'].tail(500).plot()
    # dfreg['Poly2'].tail(500).plot()
    # dfreg['Poly3'].tail(500).plot()
    # dfreg['Knn'].tail(500).plot()
    #
    # plt.legend(loc=4)
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.show()
    return dfreg['Adj Close'].dropna().to_json(), dfreg['Linear'].dropna().to_json(), \
           dfreg['Poly2'].dropna().to_json(), dfreg['Poly3'].dropna().to_json(), \
           dfreg['Knn'].dropna().to_json()


if __name__ == '__main__':
    predict_stock_price()
