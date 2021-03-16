import datetime
import glob
import math
import os
from pathlib import Path
from urllib.request import urlopen, Request
import matplotlib as mpl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
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

def get_finviz_data():
    finviz_url = 'https://finviz.com/quote.ashx?t='
    stocks = ['AMZN', 'GOOG', 'FB']

    news_tables = {}
    for stock in stocks:
        url = finviz_url + stock

        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)

        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[stock] = news_table

def parse_and_analyze():
    parsed_data = []

    for stock, news_table in news_tables.items():

        for row in news_table.findAll('tr'):

            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([stock, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['stock', 'date', 'time', 'title'])

    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date


def plot_graph():
    plt.figure(figsize=(12,8))
    mean_df = df.groupby(['stock', 'date']).mean().unstack()
    mean_df = mean_df.xs('compound', axis="columns")
    plt.style.use("ggplot")

    amz = mean_df.loc["AMZN"]
    fb = mean_df.loc["FB"]
    goog = mean_df.loc["GOOG"]
    amz.drop(amz[amz.isna()==True].index, inplace=True)
    xtickspos = []
    for i in range(len(amz.values)):
        xtickspos.append(i+2.5)
    for i in range(len(fb.values)):
        xtickspos.append(i+9)
    for i in range(len(goog.values)):
        xtickspos.append(i+19.5)
    plt.bar(xtickspos[0: len(amz.values)] , amz.values)
    plt.bar(xtickspos[len(amz.values): len(fb.values)+len(amz.values)] , fb.values)
    plt.bar(xtickspos[len(fb.values)+len(amz.values):] , goog.values)
    plt.xticks(xtickspos, np.concatenate((amz.index, fb.index, goog.index)), rotation=90)
    plt.legend(["AMZN", "FB", "GOOG"])
    plt.show()

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
