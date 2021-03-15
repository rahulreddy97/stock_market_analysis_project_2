import io
import random
from flask import Flask, Response, request
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.figure import Figure


app = Flask(__name__)


@app.route("/")
def index():
    """ Returns html with the img tag for your plot.
    """
    num_x_points = int(request.args.get("num_x_points", 50))
    # in a real app you probably want to use a flask template.
    return f"""
    <h1>Flask and matplotlib</h1>
    <h2>Random data with num_x_points={num_x_points}</h2>
    <form method=get action="/">
      <input name="num_x_points" type=number value="{num_x_points}" />
      <input type=submit value="update graph">
    </form>
    <h3>Plot as a png</h3>
    <img src="/matplot-as-image-{num_x_points}.png"
         alt="random points as png"
         height="200"
    >
    <h3>Plot as a SVG</h3>
    <img src="/matplot-as-image-{num_x_points}.svg"
         alt="random points as svg"
         height="200"
    >
    """
    # from flask import render_template
    # return render_template("yourtemplate.html", num_x_points=num_x_points)


@app.route("/matplot-as-image-<int:num_x_points>.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/matplot-as-image-<int:num_x_points>.svg")
def plot_svg(num_x_points=50):
    """ renders the plot on the fly.
    """
    finviz_url = 'https://finviz.com/quote.ashx?t='
    tickers = ['AMZN', 'GOOG', 'FB']

    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker

        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)

        html = BeautifulSoup(response, features='html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    parsed_data = []

    for ticker, news_table in news_tables.items():

        for row in news_table.findAll('tr'):

            title = row.a.text
            date_data = row.td.text.split(' ')

            if len(date_data) == 1:
                time = date_data[0]
            else:
                date = date_data[0]
                time = date_data[1]

            parsed_data.append([ticker, date, time, title])

    df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

    vader = SentimentIntensityAnalyzer()

    f = lambda title: vader.polarity_scores(title)['compound']
    df['compound'] = df['title'].apply(f)
    df['date'] = pd.to_datetime(df.date).dt.date

    plt.figure(figsize=(10,8))
    mean_df = df.groupby(['ticker', 'date']).mean().unstack()
    mean_df = mean_df.xs('compound', axis="columns")
    mean_df.plot(kind='bar')
    plt.show()
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")


if __name__ == "__main__":
    import webbrowser

    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)