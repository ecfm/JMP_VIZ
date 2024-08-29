import json
import os
from collections import defaultdict

import dash
import numpy as np
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output, State

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT_DIR, "result/MRO_silicon_tube")

def merge_values(val1, val2):
    if val1 is None or len(val1) == 0:
        return val2
    if val2 is None or len(val2) == 0:
        return val1
    if isinstance(val1, list) or isinstance(val2, list):
        if isinstance(val1, list) and isinstance(val2, list):
            return list(set(val1+val2))
        elif hasattr(val1, 'keys'):
            return merge_values(val1, {'_': val2})
        else:
            return merge_values({'_': val1}, val2)
    elif hasattr(val1, 'keys') and hasattr(val2, 'keys'):
        new_dict = {}
        for key in set(list(val1.keys())+list(val2.keys())):
            new_dict[key] = merge_values(val1.get(key, None), val2.get(key, None))
        return new_dict
    else:
        raise ValueError(f"Invalid value type: {type(val1)} of {val1} and {type(val2)} of {val2}")

def json_paths_to_string(json_input):
    if not isinstance(json_input, dict):
        return {}
    result = {}
    for key, val in json_input.items():
        key_words = key.replace('_', ' ')
        if isinstance(val, dict):
            for k, v in val.items():
                result[f"{key_words}||{k}"] = v
        else:
            result[key_words] = val
    return result

def calculate_mentions(id_to_values):
    mentions = {}
    for id, values in id_to_values.items():
        mentions[id] = len(values)
    return mentions

def calculate_co_mentions(use_id_to_values, attr_perf_dict):
    co_mentions = defaultdict(lambda: defaultdict(int))
    for use_id, use_values in use_id_to_values.items():
        for attr_or_perf_id in attr_perf_dict.get('attr', {}).keys():
            if attr_or_perf_id in use_values:
                co_mentions[use_id][attr_or_perf_id] = len(set(use_values[attr_or_perf_id]))
        for attr_or_perf_id in attr_perf_dict.get('perf', {}).keys():
            if attr_or_perf_id in use_values:
                co_mentions[use_id][attr_or_perf_id] = len(set(use_values[attr_or_perf_id]))
    return co_mentions

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict):
    matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    sentiment_matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    review_matrix = np.empty((len(y_path_to_ids_dict), len(x_path_to_sents_dict)), dtype=object)
    review_matrix.fill([])
    for i, (path1, y_ids) in enumerate(y_path_to_ids_dict.items()):
        for j, (path2, x_sents_dict) in enumerate(x_path_to_sents_dict.items()):
            x_sent_ids = []
            for sent, reason_rids in x_sents_dict.items():
                x_sent_ids.extend(reason_rids.keys())
            x_sent_ids = set(x_sent_ids)
            co_mention_ids = (set(y_ids)).intersection(x_sent_ids)
            co_mentions = 0
            pos_mentions = 0
            for sent, reason_rids in x_sents_dict.items():
                for cm_id in co_mention_ids:
                    if cm_id in reason_rids:
                        reviews = reason_rids[cm_id]
                        review_matrix[i, j] = review_matrix[i, j] +reviews
                        co_mentions += len(reviews)
                        if sent == '+':
                            pos_mentions += len(reviews)
            matrix[i, j] = co_mentions
            if co_mentions > 0:
                sentiment_matrix[i, j] = pos_mentions / co_mentions
    return matrix, sentiment_matrix, review_matrix

def ratio_to_rgb(ratio):
    ratio = max(0.00001, min(ratio, 0.99999))
    rgba = plotly.colors.sample_colorscale(px.colors.diverging.RdBu, ratio)[0]
    return rgba

app = dash.Dash(__name__)
server = app.server  # Add this line
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('Plot Type:'),
            dcc.RadioItems(
                id='plot-type',
                options=[
                    {'label': 'Use v.s. Attribute + Performance', 'value': 'use_attr_perf'},
                    {'label': 'Performance v.s. Attribute', 'value': 'perf_attr'}
                ],
                value='use_attr_perf',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
        ], style={'width': '100%', 'display': 'inline-block', 'margin-bottom': '10px'}),
        html.Div([
            html.Label('Y-axis Category:'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': 'All [Level 0]', 'value': 'all'}],
                value='all',
                placeholder="Select a category for Y-axis"
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Label('X-axis Category:'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': 'All [Level 0]', 'value': 'all'}],
                value='all',
                placeholder="Select a category for X-axis"
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),
    html.Div([
        dcc.Graph(id='correlation-matrix', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}),
        html.Div([
            html.H3("Reviews"),
            html.Div(id='reviews-content', style={'maxHeight': '300px', 'overflowY': 'auto', 'border': '2px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
        ])
    ], style={'borderTop': '3px double #ddd'})
])

def filter_dict(path_to_sents_dict, category):
    if category == 'all':
        key_part_num = 1
    else:
        key_part_num = len(category.split('|'))+2
    filtered_dict = defaultdict(dict)
    for path, sents_dict in path_to_sents_dict.items():
        if category == 'all' or path.startswith(category):
            key = '|'.join(path.split('|')[:key_part_num])
            filtered_dict[key] = merge_values(filtered_dict[key], sents_dict)
    return filtered_dict

TOP_N = 7

use_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'use_path_to_sents_dict.json')))
attr_perf_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_ids_dict.json')))
perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'perf_path_to_sents_dict.json')))
attr_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_ids_dict.json')))

path_dict_cache = {'use_sents': (defaultdict(dict), use_path_to_sents_dict), 
'perf_sents': (defaultdict(dict), perf_path_to_sents_dict), 
'attr_ids': (defaultdict(dict), attr_path_to_ids_dict), 
'attr_perf_ids': (defaultdict(dict), attr_perf_path_to_ids_dict)}

def get_cached_dict(type, category):
    dict_cache, raw_dict = path_dict_cache[type]
    if category in dict_cache:
        return dict_cache[category]
    else:
        filtered_dict = filter_dict(raw_dict, category)
        dict_cache[category] = filtered_dict
        return filtered_dict

plot_data_cache = {}

def get_plot_data(plot_type, x_category='all', y_category='all'):
    plot_key = f'{plot_type}##{x_category}##{y_category}'
    if plot_key in plot_data_cache:
        return plot_data_cache[plot_key]
    if plot_type == 'use_attr_perf':
        x_path_to_sents_dict = get_cached_dict('use_sents', x_category)
        y_path_to_ids_dict = get_cached_dict('attr_perf_ids', y_category)
        title = 'Correlation Matrix of Silicone Tube Usage v.s. Product Performance and Attributes'
    else:  # perf_attr
        x_path_to_sents_dict = get_cached_dict('perf_sents', x_category)
        y_path_to_ids_dict = get_cached_dict('attr_ids', y_category)
        title = 'Correlation Matrix of Silicone Tube Performance v.s. Attributes'

    if x_category == 'all':
        x_path_to_sents_dict = x_path_to_sents_dict
    else:
        x_path_to_sents_dict = {k: v for k, v in x_path_to_sents_dict.items() if k.startswith(x_category)}

    if y_category == 'all':
        y_path_to_ids_dict = y_path_to_ids_dict
    else:
        y_path_to_ids_dict = {k: v for k, v in y_path_to_ids_dict.items() if k.startswith(y_category)}
    matrix, sentiment_matrix, review_matrix = create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict)
    # identify top x path indeces from summing up the matrix
    top_x_indices = np.argsort(np.sum(matrix, axis=0))[-TOP_N:]
    top_x_paths = [list(x_path_to_sents_dict.keys())[i] for i in top_x_indices]
    # identify top y path indeces with non-zero sum from summing up the matrix
    y_sums = np.sum(matrix, axis=1)
    non_zero_y = np.where(y_sums > 0)[0]
    top_y_indices = non_zero_y[np.argsort(y_sums[non_zero_y])[-TOP_N:]]
    top_y_paths = [list(y_path_to_ids_dict.keys())[i] for i in top_y_indices]
    
    matrix = matrix[top_y_indices, :][:, top_x_indices]
    sentiment_matrix = sentiment_matrix[top_y_indices, :][:, top_x_indices]
    review_matrix = review_matrix[top_y_indices, :][:, top_x_indices]

    if x_category == 'all':
        x_text = top_x_paths
    else:
        x_text = [path[len(x_category)+1:] for path in top_x_paths]
    if y_category == 'all':
        y_text = top_y_paths
    else:
        y_text = [path[len(y_category)+1:] for path in top_y_paths]
    plot_data_cache[plot_key] = (matrix, sentiment_matrix, review_matrix, x_text, y_text, title)
    return plot_data_cache[plot_key]

def get_options(value, top_paths):
    levels = [{'label': 'All [Level 0]', 'value': 'all'}]
    if value != 'all':
        parts = value.split('|')
        for i, part in enumerate(parts):
            current_val = '|'.join(parts[:i+1])
            levels.append({'label': f'{current_val} [Level {i+1}]', 'value': current_val})
        remained_paths = [{'label': path+f' [Level {len(parts)+1}]', 'value': path} for path in top_paths if (path.startswith(value) and path != value)]
        if len(remained_paths) > 0:
            return levels + remained_paths
        else:
            return levels
    else:
        return levels + [{'label': path+' [Level 1]', 'value': path} for path in top_paths]


@app.callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('x-axis-dropdown', 'value'),
     Output('y-axis-dropdown', 'value'),
     Output('correlation-matrix', 'figure')],
    [Input('plot-type', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_graph(plot_type, x_value, y_value):
    # Get data for the graph
    matrix, sentiment_matrix, review_matrix, x_text, y_text, title = get_plot_data(plot_type, x_value, y_value)
    
    # Create the heatmap figure
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=x_text,
        y=y_text,
        colorscale=[[0, 'rgba(255,255,255,0)'], [1, 'rgba(255,255,255,0)']],
        showscale=False,
    ))

    max_mentions = np.max(matrix)

    for i in range(len(y_text)):
        for j in range(len(x_text)):
            if matrix[i, j] > 0:
                box_width = 0.8 * matrix[i, j] / max_mentions
                fig.add_shape(
                    type="rect",
                    x0=j-box_width/2, y0=i-0.4, x1=j+box_width/2, y1=i+0.4,
                    fillcolor=ratio_to_rgb(sentiment_matrix[i, j]),
                    line_color="rgba(0,0,0,0)",
                )

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45, tickmode='array', tickvals=list(range(len(x_text))), ticktext=x_text),
        yaxis=dict(tickmode='array', tickvals=list(range(len(y_text))), ticktext=y_text),
        width=1500,
        height=1000,
    )

    fig.update_layout(
        coloraxis=dict(
            colorbar=dict(
                title="Satisfaction Level",
                tickvals=[0, 0.5, 1],
                ticktext=["Unsatisfied", "Neutral", "Satisfied"],
            ),
            colorscale=px.colors.diverging.RdBu,
        )
    )

    fig.update_traces(
        hovertemplate="Y: %{y}<br>X: %{x}<br>Count: %{z}<br>Satisfaction Ratio: %{customdata:.2f}",
        customdata=sentiment_matrix
    )
    x_options = get_options(x_value, x_text)
    y_options = get_options(y_value, y_text)
    
    return x_options, y_options, x_value, y_value, fig

@app.callback(
    Output('reviews-content', 'children'),
    [Input('correlation-matrix', 'clickData')],
    [State('plot-type', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value')]
)
def display_clicked_reviews(click_data, plot_type, x_value, y_value):
    if click_data:
        matrix, sentiment_matrix, review_matrix, x_text, y_text, title = get_plot_data(plot_type, x_value, y_value)
        point = click_data['points'][0]
        i, j = y_text.index(point['y']), x_text.index(point['x'])
        reviews = review_matrix[i][j]
        selected_x = x_text[j]
        selected_y = y_text[i]
        content = [
            dcc.Markdown(f"Selected: X=<b>{selected_x}</b>, Y=<b>{selected_y}</b>", dangerously_allow_html=True)
        ]
        if reviews:
            reviews_text = "<br>".join([f"Review {idx + 1}: {review}" for idx, review in enumerate(set(reviews))])
            content.append(dcc.Markdown(reviews_text, dangerously_allow_html=True))
        else:
            content.append(html.P("No reviews available for this selection."))
        return content
    return []

if __name__ == '__main__':
    app.run_server(debug=True)