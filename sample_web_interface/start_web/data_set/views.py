import django
import io
from django.shortcuts import render
from django.http import HttpResponse
import sample_web_interface.start_web.ml_core.main as ml_core
import sample_web_interface.start_web.statistic_core as stat_core
import sample_web_interface.start_web.visualization_core.main as vis_core

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


def index(request):
    context = {
        'data': ml_core.hello_world()
    }
    return render(request, "data_set/model_training.html", context)


def model_test(request):
    return render(request, "data_set/model_test.html")


def make_plot(request):
    fig = vis_core.get_plot([],[])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    response = HttpResponse(buf.getvalue(), content_type='image/png')
    return response

