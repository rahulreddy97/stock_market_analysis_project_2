from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

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

plt.figure(figsize=(10,8))
mean_df = df.groupby(['stock', 'date']).mean().unstack()
mean_df = mean_df.xs('compound', axis="columns")
mean_df.plot(kind='bar')
plt.show()

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
