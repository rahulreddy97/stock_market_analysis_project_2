import datetime
from wsgiref.simple_server import WSGIServer

from flask import Flask, render_template, request

from stock_predict import predict_stock_price

app = Flask(__name__)


def to_date(date_string):
    return datetime.datetime.strptime(date_string, "%Y-%m-%d").date()


@app.route('/', methods=["GET", "POST"])
def stock_prediction():
    comp = request.args.get('comp', default='AAPL', type=str)
    if len(comp) == 0:
        comp = 'AAPL'
    start_date = to_date(request.args.get('sdate', default=(
            datetime.datetime.now() - datetime.timedelta(days=3 * 365)).date().isoformat()))
    end_date = to_date(request.args.get('edate', default=datetime.date.today().isoformat()))

    plot_data, linear_data, poly2_data, poly3_data, knn_data = predict_stock_price(comp, start_date, end_date)

    return render_template('index.html', comp=comp, plot_data=plot_data, linear_data=linear_data,
                           poly2_data=poly2_data,
                           poly3_data=poly3_data, knn_data=knn_data)

# def stock_prediction2():

if __name__ == '__main__':
    # app.config["DEBUG"] = True
    app.run(debug=True, host='0.0.0.0', port=5000)
    # TODO: switch to wsgi
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
