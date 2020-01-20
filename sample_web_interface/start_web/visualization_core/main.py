import pandas as pd
import matplotlib.pyplot as plt

import io
import urllib, base64

def get_plot(data_x, data_y):
    fig, ax = plt.subplots()
    ax.scatter(data_x, data_y)

    """
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    """