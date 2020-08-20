import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def get_plot_2(model, data, model_columns, train_column):
    X = data.loc[:, model_columns].values
    date = data.iloc[:, 0].values.reshape(-1)
    y_range = model.predict(X)

    fig = px.scatter(data, x=data.columns[0], y=train_column, opacity=0.5)
    fig.add_traces(go.Scatter(x=date, y=y_range, name=train_column + " prediction"))

    return fig
