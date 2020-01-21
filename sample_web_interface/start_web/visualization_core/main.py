import pandas as pd
import matplotlib.pyplot as plt

import io
import urllib, base64

def get_plot(data_x, data_y):
    fig, ax = plt.subplots()
    data_x = [1, 2, 3, 4,5,6,7,8,9]
    data_y = [1, 2, 3, 4,3,3,4,2,1]
    ax.set_xlabel('ax1')
    ax.set_ylabel('ax2')
    plt.plot(data_x, data_y)
    return fig


    """
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    """