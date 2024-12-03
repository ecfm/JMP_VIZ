import json
import os
from collections import defaultdict
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.colors
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user

# Constants and directory setup
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(ROOT_DIR, "result/MRO_cables")

# Language translations
TRANSLATIONS = {
    'en': {
        'login': 'Login',
        'username': 'Username',
        'password': 'Password',
        'enter_username': 'Enter username',
        'enter_password': 'Enter password',
        'invalid_credentials': 'Invalid credentials',
        'logout': 'Logout',
        'plot_type': 'Plot Type:',
        'use_vs_attr_perf': 'Use v.s. Attribute + Performance',
        'perf_vs_attr': 'Performance v.s. Attribute',
        'y_axis_category': 'Y-axis Category:',
        'x_axis_category': 'X-axis Category:',
        'num_y_features': 'Number of Y Features:',
        'num_x_features': 'Number of X Features:',
        'all_level_0': 'All [Level 0]',
        'reviews': 'Reviews',
        'no_reviews': 'No reviews available for this selection.',
        'selected': 'Selected:',
        'review': 'Review',
        'satisfaction_level': 'Satisfaction Level',
        'unsatisfied': 'Unsatisfied',
        'neutral': 'Neutral',
        'satisfied': 'Satisfied',
        'use_attr_perf_title': 'Usage v.s. Product Performance and Attributes',
        'perf_attr_title': 'Performance v.s. Attributes',
        'review_format': 'Display Format: Review x: ASIN <b>Review Title</b> Review Text'
    },
    'zh': {
        'login': '登录',
        'username': '用户名',
        'password': '密码',
        'enter_username': '请输入用户名',
        'enter_password': '请输入密码',
        'invalid_credentials': '用户名或密码错误',
        'logout': '退出登录',
        'plot_type': '图表类型：',
        'use_vs_attr_perf': '用途 vs. 属性 + 性能',
        'perf_vs_attr': '性能 vs. 属性',
        'y_axis_category': 'Y轴类别：',
        'x_axis_category': 'X轴类别：',
        'num_y_features': 'Y特征数量：',
        'num_x_features': 'X特征数量：',
        'all_level_0': '全部 [层级 0]',
        'reviews': '评论',
        'no_reviews': '该选择没有可用的评论。',
        'selected': '已选择：',
        'review': '评论',
        'satisfaction_level': '满意度',
        'unsatisfied': '不满意',
        'neutral': '中性',
        'satisfied': '满意',
        'use_attr_perf_title': '用途 vs. 产品属性和性能',
        'perf_attr_title': '性能 vs. 属性',
        'review_format': '显示格式： 评论 x: ASIN <b>评论标题</b> 评论内容'
    }
}

# Authentication configuration
VALID_USERNAME = "xsight"
VALID_PASSWORD = "mit100"

class User(UserMixin):
    def __init__(self, username):
        self.id = username

# Helper functions
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

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict):
    matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    sentiment_matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    review_matrix = np.empty((len(y_path_to_ids_dict), len(x_path_to_sents_dict)), dtype=object)
    
    # Initialize each cell with its own empty list
    for i in range(review_matrix.shape[0]):
        for j in range(review_matrix.shape[1]):
            review_matrix[i, j] = []
    
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
                        review_matrix[i, j].extend(reviews)  # Extend instead of assign
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

# Load data
use_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'use_path_to_sents_dict.json')))
attr_perf_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_ids_dict.json')))
perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'perf_path_to_sents_dict.json')))
attr_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_ids_dict.json')))

path_dict_cache = {
    'use_sents': (defaultdict(dict), use_path_to_sents_dict),
    'perf_sents': (defaultdict(dict), perf_path_to_sents_dict),
    'attr_ids': (defaultdict(dict), attr_path_to_ids_dict),
    'attr_perf_ids': (defaultdict(dict), attr_perf_path_to_ids_dict)
}

# Initialize Dash app and login manager
app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# Add these secret key settings after app initialization:
server.config.update(
    SECRET_KEY='your-secret-key-here',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True if os.environ.get('PRODUCTION', False) else False,
)

login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

@login_manager.user_loader
def load_user(username):
    if username == VALID_USERNAME:
        return User(username)
    return None

# Layout definitions
login_layout = html.Div([
    html.H2('Login', style={'textAlign': 'center', 'marginTop': '50px'}),
    html.Div([
        html.Div([
            html.Label('Username'),
            dcc.Input(
                id='username-input',
                type='text',
                placeholder='Enter username',
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            html.Label('Password'),
            dcc.Input(
                id='password-input',
                type='password',
                placeholder='Enter password',
                style={'width': '100%', 'marginBottom': '10px'}
            ),
            html.Button('Login', id='login-button', n_clicks=0),
            html.Div(id='login-error')
        ], style={
            'width': '300px',
            'margin': '0 auto',
            'padding': '20px',
            'border': '1px solid #ddd',
            'borderRadius': '5px'
        })
    ])
])

main_layout = html.Div([
    html.Div([
        html.Div([
        html.Button('Logout', id='logout-button', style={'float': 'right', 'margin': '10px'}),
    ]),
        html.Div([
            html.Label(id='plot-type-label'),
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
            html.Div([
            html.Label(id='y-axis-label'),  # Changed to use ID
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': 'All [Level 0]', 'value': 'all'}],
                value='all',
                placeholder="Select a category for Y-axis"
            ),
            html.Label(id='num-y-features-label', style={'marginTop': '10px'}),  # Changed to use ID
                dcc.Slider(
                    id='y-features-slider',
                    min=1,
                    max=10,
                    value=7,
                    step=1,  # Ensure integer steps
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
            html.Label(id='x-axis-label'), 
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': 'All [Level 0]', 'value': 'all'}],
                value='all',
                placeholder="Select a category for X-axis"
            ),
            html.Label(id='num-x-features-label', style={'marginTop': '10px'}),
                dcc.Slider(
                    id='x-features-slider',
                    min=1,
                    max=10,
                    value=7,
                    step=1,  # Ensure integer steps
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
        ]),
    ]),
    html.Div([
        dcc.Graph(id='correlation-matrix', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}),
        html.Div([
            html.H3("Reviews"),
            html.Div(id='reviews-content', style={'maxHeight': '300px', 'overflowY': 'auto', 'border': '2px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
        ])
    ], style={'borderTop': '3px double #ddd'})
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Move language selector to upper left and adjust styling
    html.Div([
        dcc.RadioItems(
            id='language-selector',
            options=[
                {'label': 'English', 'value': 'en'},
                {'label': '中文', 'value': 'zh'}
            ],
            value='zh',
            style={'float': 'left', 'margin': '10px'}  # Changed from 'right' to 'left'
        ),
    ], style={'width': '100%', 'clear': 'both'}),  # Added container div with clear
    html.Div(id='page-content')
])

# Data processing functions
def get_cached_dict(type, category):
    dict_cache, raw_dict = path_dict_cache[type]
    if category in dict_cache:
        return dict_cache[category]
    else:
        filtered_dict = filter_dict(raw_dict, category)
        dict_cache[category] = filtered_dict
        return filtered_dict

plot_data_cache = {}

def get_plot_data(plot_type, x_category='all', y_category='all', top_n_x=2, top_n_y=2):
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    plot_key = f'{plot_type}##{x_category}##{y_category}##{top_n_x}##{top_n_y}'
    if plot_key in plot_data_cache:
        return plot_data_cache[plot_key]
    
    if plot_type == 'use_attr_perf':
        x_path_to_sents_dict = get_cached_dict('use_sents', x_category)
        y_path_to_ids_dict = get_cached_dict('attr_perf_ids', y_category)
        title_key = 'use_attr_perf_title'
    else:  # perf_attr
        x_path_to_sents_dict = get_cached_dict('perf_sents', x_category)
        y_path_to_ids_dict = get_cached_dict('attr_ids', y_category)
        title_key = 'perf_attr_title'

    matrix, sentiment_matrix, review_matrix = create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict)
    
    # Ensure we don't exceed the available features
    top_n_x = min(top_n_x, matrix.shape[1])
    top_n_y = min(top_n_y, matrix.shape[0])
    
    # identify top x path indices from summing up the matrix
    top_x_indices = np.argsort(np.sum(matrix, axis=0))[-top_n_x:]
    top_x_paths = [list(x_path_to_sents_dict.keys())[i] for i in top_x_indices]
    
    # identify top y path indices with non-zero sum
    y_sums = np.sum(matrix, axis=1)
    non_zero_y = np.where(y_sums > 0)[0]
    if len(non_zero_y) == 0:  # Handle case with no non-zero values
        top_y_indices = np.arange(min(top_n_y, matrix.shape[0]))
    else:
        top_y_indices = non_zero_y[np.argsort(y_sums[non_zero_y])[-min(top_n_y, len(non_zero_y)):]]
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
        
    plot_data_cache[plot_key] = (matrix, sentiment_matrix, review_matrix, x_text, y_text, title_key) 
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
    [Output('url', 'pathname'),
     Output('login-error', 'children')],
    [Input('login-button', 'n_clicks'),
     Input('username-input', 'value'),
     Input('password-input', 'value')],
    prevent_initial_call=True
)
def login_callback(n_clicks, username, password):
    if not n_clicks:
        return dash.no_update, dash.no_update
    
    if username == VALID_USERNAME and password == VALID_PASSWORD:
        user = User(username)
        login_user(user)
        return '/', ''
    return dash.no_update, html.Div('Invalid credentials', style={'color': 'red'})

@app.callback(
    [Output('plot-type-label', 'children'),
     Output('plot-type', 'options')],
    [Input('language-selector', 'value')]
)  # Added missing closing parenthesis
def update_plot_type_labels(language):
    options = [
        {'label': TRANSLATIONS[language]['use_vs_attr_perf'], 'value': 'use_attr_perf'},
        {'label': TRANSLATIONS[language]['perf_vs_attr'], 'value': 'perf_attr'}
    ]
    return TRANSLATIONS[language]['plot_type'], options

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('language-selector', 'value')]
)
def display_page(pathname, language='en'):
    if pathname == '/login' or not current_user.is_authenticated:
        return html.Div([
            html.H2(TRANSLATIONS[language]['login'], style={'textAlign': 'center', 'marginTop': '50px'}),
            html.Div([
                html.Div([
                    
                    html.Label(TRANSLATIONS[language]['username']),
                    dcc.Input(
                        id='username-input',
                        type='text',
                        placeholder=TRANSLATIONS[language]['enter_username'],
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    html.Label(TRANSLATIONS[language]['password']),
                    dcc.Input(
                        id='password-input',
                        type='password',
                        placeholder=TRANSLATIONS[language]['enter_password'],
                        style={'width': '100%', 'marginBottom': '10px'}
                    ),
                    html.Button(TRANSLATIONS[language]['login'], id='login-button', n_clicks=0),
                    html.Div(id='login-error')
                ], style={
                    'width': '300px',
                    'margin': '0 auto',
                    'padding': '20px',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px'
                })
            ])
        ])
    if pathname == '/':
        return main_layout
    return login_layout

@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    [Input('logout-button', 'n_clicks')],
    [State('url', 'pathname')],
    prevent_initial_call=True
)
def logout_callback(n_clicks, pathname):
    if n_clicks is not None and n_clicks > 0:
        logout_user()
        return '/login'
    return pathname

# Add new callback to update the labels
@app.callback(
    [Output('y-axis-label', 'children'),
     Output('x-axis-label', 'children'),
     Output('num-y-features-label', 'children'),
     Output('num-x-features-label', 'children')],
    [Input('language-selector', 'value')]
)
def update_axis_labels(language):
    return (
        TRANSLATIONS[language]['y_axis_category'],
        TRANSLATIONS[language]['x_axis_category'],
        TRANSLATIONS[language]['num_y_features'],
        TRANSLATIONS[language]['num_x_features']
    )

@app.callback(
    [Output('x-axis-dropdown', 'options'),
     Output('y-axis-dropdown', 'options'),
     Output('x-axis-dropdown', 'value'),
     Output('y-axis-dropdown', 'value'),
     Output('correlation-matrix', 'figure'),
     Output('x-features-slider', 'max'),
     Output('y-features-slider', 'max')],
    [Input('plot-type', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('x-features-slider', 'value'),
     Input('y-features-slider', 'value'),
     Input('language-selector', 'value')]
)
def update_graph(plot_type, x_value, y_value, top_n_x, top_n_y, language):
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    # Get data for the graph
    if plot_type == 'use_attr_perf':
        x_dict = get_cached_dict('use_sents', x_value)
        y_dict = get_cached_dict('attr_perf_ids', y_value)
    else:
        x_dict = get_cached_dict('perf_sents', x_value)
        y_dict = get_cached_dict('attr_ids', y_value)
    
    # Update slider maximums based on available features
    max_x = len(x_dict)
    max_y = len(y_dict)
    
    # Ensure top_n values don't exceed available features
    top_n_x = min(top_n_x, max_x)
    top_n_y = min(top_n_y, max_y)
    
    matrix, sentiment_matrix, review_matrix, x_text, y_text, title_key = get_plot_data(
        plot_type, x_value, y_value, top_n_x, top_n_y
    )
    
    # Calculate dynamic dimensions based on number of features
    # Height calculation
    base_height = 400
    min_height = 300
    additional_height_per_feature = 40
    dynamic_height = max(
        min_height,
        base_height + max(0, len(y_text) - 5) * additional_height_per_feature
    )
    
    # Width calculation - increase base width to accommodate legends
    base_width = 800  # Increased from 600
    min_width = 600  # Increased from 400
    additional_width_per_feature = 100
    max_width = 1600  # Increased from 1200
    
    # Calculate width based on the length of x-axis labels and number of features
    avg_label_length = sum(len(str(label)) for label in x_text) / len(x_text) if x_text else 0
    width_factor = max(1, avg_label_length / 15)  # Adjust width more for longer labels
    
    dynamic_width = min(
        max_width,
        max(
            min_width,
            base_width + max(0, len(x_text) - 5) * additional_width_per_feature * width_factor
        )
    )
    
    # Create the base heatmap figure
    fig = go.Figure()

    # Calculate dynamic margins based on label lengths
    max_y_label_length = max(len(str(label)) for label in y_text) if y_text else 0
    left_margin = min(200, max(80, max_y_label_length * 7))

    # Add the satisfaction ratio heatmap (color legend)
    fig.add_trace(go.Heatmap(
        z=sentiment_matrix,
        x=x_text,
        y=y_text,
        colorscale=px.colors.diverging.RdBu,
        showscale=True,
        opacity=0,
        colorbar=dict(
            title=TRANSLATIONS[language]['satisfaction_level'],
            tickvals=[0, 0.5, 1],
            ticktext=[
                TRANSLATIONS[language]['unsatisfied'],
                TRANSLATIONS[language]['neutral'],
                TRANSLATIONS[language]['satisfied']
            ],
            x=1.15
        )
    ))

    # Add width legend translations
    width_legend_translations = {
        'en': {
            'explanation': "Width is proportional to the number of reviews"
        },
        'zh': {
            'explanation': "宽度与评论数量成正比"
        }
    }

    # Add the custom shapes for the actual visualization
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

    # Find a good example box from the actual data
    example_i, example_j = np.unravel_index(np.argmax(matrix), matrix.shape)
    example_sentiment = sentiment_matrix[example_i, example_j]
    example_count = int(matrix[example_i, example_j])
    
    # Constants for width example box and explanation positioning
    EXAMPLE_BOX_LEFT = 1.1
    EXAMPLE_BOX_RIGHT = EXAMPLE_BOX_LEFT + 0.12
    EXAMPLE_BOX_BOTTOM = 1.1 
    EXAMPLE_BOX_TOP = EXAMPLE_BOX_BOTTOM + 0.11

    BRACKET_TOP = EXAMPLE_BOX_TOP + 0.06
    BRACKET_VERTICAL_LENGTH = 0.03  # Length of vertical bracket lines

    EXPLANATION_X = (EXAMPLE_BOX_LEFT + EXAMPLE_BOX_RIGHT) / 2 + 0.22  # Center explanation above bracket
    EXPLANATION_Y = BRACKET_TOP + 0.1  # Position explanation above bracket

    # Add explanation text with translation above the bracket
    fig.add_annotation(
        xref="paper", yref="paper", 
        x=EXPLANATION_X,
        y=EXPLANATION_Y,
        text=width_legend_translations[language]['explanation'].format(example_count),
        showarrow=False,
        font=dict(size=12),
        align="center",
        width=200
    )

    # Add the example box
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=EXAMPLE_BOX_LEFT, 
        x1=EXAMPLE_BOX_RIGHT,
        y0=EXAMPLE_BOX_BOTTOM, 
        y1=EXAMPLE_BOX_TOP,
        fillcolor=ratio_to_rgb(example_sentiment),
        line_color="rgba(0,0,0,0)",
    )

    # Add bracket at the top of the box
    fig.add_shape(
        type="line",
        xref="paper", yref="paper",
        x0=EXAMPLE_BOX_LEFT, 
        x1=EXAMPLE_BOX_RIGHT,
        y0=BRACKET_TOP, 
        y1=BRACKET_TOP,
        line=dict(color="black", width=1)
    )

    # Add vertical lines for bracket
    for x in [EXAMPLE_BOX_LEFT, EXAMPLE_BOX_RIGHT]:
        fig.add_shape(
            type="line",
            xref="paper", yref="paper",
            x0=x, x1=x,
            y0=BRACKET_TOP, 
            y1=BRACKET_TOP - BRACKET_VERTICAL_LENGTH,
            line=dict(color="black", width=1)
        )

    # Update layout with even more right margin
    fig.update_layout(
        title=TRANSLATIONS[language][title_key],
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=list(range(len(x_text))),
            ticktext=x_text,
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(y_text))),
            ticktext=y_text,
        ),
        width=dynamic_width,
        height=dynamic_height,
        margin=dict(
            l=left_margin,
            r=400,  # Increased from 300 to 400 to accommodate the explanation text
            t=100,
            b=100,
            autoexpand=True
        )
    )

    # First, add the needed translations to both language dictionaries
    hover_translations = {
        'en': {
            'hover_y': 'Y-axis',
            'hover_x': 'X-axis',
            'hover_count': 'Count',
            'hover_satisfaction': 'Satisfaction Ratio'
        },
        'zh': {
            'hover_y': 'Y轴',
            'hover_x': 'X轴',
            'hover_count': '数量',
            'hover_satisfaction': '满意度比例'
        }
    }

    # Update hover template with translations
    hover_template = (
        f"{hover_translations[language]['hover_y']}: %{{y}}<br>" +
        f"{hover_translations[language]['hover_x']}: %{{x}}<br>" +
        f"{hover_translations[language]['hover_count']}: %{{text}}<br>" +
        f"{hover_translations[language]['hover_satisfaction']}: %{{customdata:.2f}}"
    )

    fig.update_traces(
        hovertemplate=hover_template,
        customdata=sentiment_matrix,
        text=matrix
    )

    x_options = get_options(x_value, x_text)
    y_options = get_options(y_value, y_text)
    
    return x_options, y_options, x_value, y_value, fig, max_x, max_y

@app.callback(
    Output('reviews-content', 'children'),
    [Input('correlation-matrix', 'clickData'),
     Input('language-selector', 'value')],
    [State('plot-type', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('x-features-slider', 'value'),
     State('y-features-slider', 'value')]
)
def display_clicked_reviews(click_data, language, plot_type, x_value, y_value, top_n_x, top_n_y):
    if click_data:
        matrix, sentiment_matrix, review_matrix, x_text, y_text, title = get_plot_data(
            plot_type, x_value, y_value, top_n_x, top_n_y
        )
        point = click_data['points'][0]
        i, j = y_text.index(point['y']), x_text.index(point['x'])
        reviews = review_matrix[i][j]
        selected_x = x_text[j]
        selected_y = y_text[i]
        content = [
            dcc.Markdown(f"{TRANSLATIONS[language]['selected']} X=<b>{selected_x}</b>, Y=<b>{selected_y}</b>", dangerously_allow_html=True),
            dcc.Markdown(f"{TRANSLATIONS[language]['review_format']}", dangerously_allow_html=True)
        ]
        if reviews:
            reviews_text = "<br>".join([f"{TRANSLATIONS[language]['review']} {idx + 1}: {review}" for idx, review in enumerate(reviews)])
            content.append(dcc.Markdown(reviews_text, dangerously_allow_html=True))
        else:
            content.append(html.P(TRANSLATIONS[language]['no_reviews']))
        return content
    return []

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')