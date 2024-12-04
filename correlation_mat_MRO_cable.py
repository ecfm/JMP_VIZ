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
import re
from typing import List, Dict
from cachetools import TTLCache, cached

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
        'review_format': 'Display Format: Review x: ASIN (Star Rating) <b>Review Title</b> Review Text',
        'percentage_explanation': 'Percentage of total mentions',
        'x_axis_percentage': 'Percentage of total mentions',
        'y_axis_percentage': 'Percentage of total mentions',
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
        'review_format': '显示格式： 评论 x: ASIN (评论星级) <b>评论标题</b> 评论内容',
        'percentage_explanation': '占总提及次数的百分比',
        'x_axis_percentage': '(该类别被提及次数%)',
        'y_axis_percentage': '(该类别被提及次数%)',
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

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict, y_path_to_sents_dict):
    # Add N/A entries
    x_path_to_sents_dict = {**x_path_to_sents_dict, 'N/A': {}}
    y_path_to_ids_dict = {**y_path_to_ids_dict, 'N/A': []}
    y_path_to_sents_dict = {**y_path_to_sents_dict, 'N/A': {}}
    
    matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    sentiment_matrix = np.zeros((len(y_path_to_ids_dict), len(x_path_to_sents_dict)))
    review_matrix = np.empty((len(y_path_to_ids_dict), len(x_path_to_sents_dict)), dtype=object)
    
    # Initialize each cell with its own empty list
    for i in range(review_matrix.shape[0]):
        for j in range(review_matrix.shape[1]):
            review_matrix[i, j] = []
    
    # Get all unique review IDs and reviews
    all_x_ids = set()
    all_y_ids = set()
    
    for x_sents_dict in x_path_to_sents_dict.values():
        for sent, reason_rids in x_sents_dict.items():
            all_x_ids.update(reason_rids.keys())
    
    for y_ids in y_path_to_ids_dict.values():
        all_y_ids.update(y_ids)
    
    # Process matrix including N/A
    for i, ((path1, y_ids), y_sents_dict) in enumerate(zip(y_path_to_ids_dict.items(), y_path_to_sents_dict.values())):
        for j, (path2, x_sents_dict) in enumerate(x_path_to_sents_dict.items()):
            if path1 == 'N/A' and path2 == 'N/A':
                continue
                
            x_sent_ids = set()
            unique_reviews = set()  # Track unique reviews
            unique_pos_reviews = set()  # Track unique positive reviews
            
            # First collect all reviews and sentiments for this x category
            for sent, reason_rids in x_sents_dict.items():
                x_sent_ids.update(reason_rids.keys())
                for rid, reviews in reason_rids.items():
                    if rid in y_ids:  # Co-mention case
                        for review in reviews:
                            unique_reviews.add(review)
                            if sent == '+':
                                unique_pos_reviews.add(review)
            
            # Handle N/A cases
            if path1 == 'N/A':
                # Reviews that mention X but not any Y
                unique_x_only = x_sent_ids - all_y_ids
                for sent, reason_rids in x_sents_dict.items():
                    for rid, reviews in reason_rids.items():
                        if rid in unique_x_only:
                            for review in reviews:
                                unique_reviews.add(review)
                                if sent == '+':
                                    unique_pos_reviews.add(review)
            
            elif path2 == 'N/A':
                # Reviews that mention Y but not any X
                # Use y_sents_dict to get sentiment information
                for sent, reviews in y_sents_dict.items():
                    for review in reviews:
                        if review not in unique_reviews:  # Only add if not already counted
                            unique_reviews.add(review)
                            if sent == '+':
                                unique_pos_reviews.add(review)
            
            # Store reviews in review matrix
            review_matrix[i, j] = list(unique_reviews)
            
            # Update counts and sentiment
            num_unique_reviews = len(unique_reviews)
            matrix[i, j] = num_unique_reviews
            
            if num_unique_reviews > 0:
                sentiment_matrix[i, j] = len(unique_pos_reviews) / num_unique_reviews
                
    return matrix, sentiment_matrix, review_matrix

def ratio_to_rgb(ratio):
    ratio = max(0.00001, min(ratio, 0.99999))
    rgba = plotly.colors.sample_colorscale(px.colors.diverging.RdBu, ratio)[0]
    return rgba

# Load data
use_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'use_path_to_sents_dict.json')))
attr_perf_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_ids_dict.json')))
attr_perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_perf_path_to_sents_dict.json')))
perf_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'perf_path_to_sents_dict.json')))
attr_path_to_ids_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_ids_dict.json')))
attr_path_to_sents_dict = json.load(open(os.path.join(RESULT_DIR, 'attr_path_to_sents_dict.json')))

# Add new cache configurations
CACHE_TTL = 3600  # Cache timeout in seconds (1 hour)
CACHE_MAXSIZE = 100  # Maximum number of items in cache

# Create cache instances
path_dict_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)
plot_data_cache = TTLCache(maxsize=CACHE_MAXSIZE, ttl=CACHE_TTL)

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

# Add new UI elements to main_layout
search_box = html.Div([
    html.Div([
        dcc.Input(
            id='search-input',
            type='text',
            placeholder='e.g. (good|great)&quality or quality&(durable|strong)',
            style={
                'width': '70%', 
                'marginRight': '10px',
                'padding': '8px',
                'fontSize': '14px'
            }
        ),
        html.Button(
            'Search', 
            id='search-button', 
            n_clicks=0,
            style={
                'padding': '8px 16px',
                'fontSize': '14px',
                'cursor': 'pointer'
            }
        ),
    ]),
    html.Div([
        html.Div(id='search-examples', style={
            'marginTop': '8px', 
            'fontSize': '12px',
            'color': '#666'
        })
    ])
], style={'marginBottom': '20px'})


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
        search_box,
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

# Map type to raw dictionaries
raw_dict_map = {
    'use_sents': use_path_to_sents_dict,
    'perf_sents': perf_path_to_sents_dict,
    'attr_sents': attr_path_to_sents_dict,
    'attr_perf_sents': attr_perf_path_to_sents_dict,
    'attr_ids': attr_path_to_ids_dict,
    'attr_perf_ids': attr_perf_path_to_ids_dict
}

# Data processing functions
@cached(cache=path_dict_cache)
def get_cached_dict(type: str, category: str, search_query: str = '', return_both: bool = False) -> Dict:
    """
    Get filtered dictionary from cache. Can return both sents and ids dicts for attr/attr_perf.
    """
    raw_dict = raw_dict_map[type]
    
    # Filter by category
    filtered_dict = filter_dict(raw_dict, category)
    
    # Only apply search query filter to sents dictionaries
    if search_query and search_query.strip() and not type.endswith('_ids'):
        filtered_dict = filter_dict_by_query(filtered_dict, search_query)
        
    # If we need both sents and ids dicts
    if return_both:
        if type == 'attr_perf_sents':
            ids_dict = filter_dict(attr_perf_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        elif type == 'attr_sents':
            ids_dict = filter_dict(attr_path_to_ids_dict, category)
            # Filter ids_dict to match paths in filtered sents dict
            ids_dict = {path: ids for path, ids in ids_dict.items() 
                       if path in filtered_dict}
        else:
            raise ValueError(f"Cannot return both dicts for type: {type}")
            
        return filtered_dict, ids_dict
        
    return filtered_dict

@cached(cache=plot_data_cache)
def get_plot_data(plot_type, x_category='all', y_category='all', top_n_x=2, top_n_y=2, language='en', search_query=''):
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    # Include search_query in cache key only if it's not empty
    cache_key = (plot_type, x_category, y_category, top_n_x, top_n_y, language, search_query) if search_query else \
                (plot_type, x_category, y_category, top_n_x, top_n_y, language)
        
    if cache_key in plot_data_cache:
        return plot_data_cache[cache_key]
    
    if plot_type == 'use_attr_perf':
        x_path_to_sents_dict = get_cached_dict('use_sents', x_category, search_query)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_perf_sents', y_category, search_query, return_both=True
        )
        title_key = 'use_attr_perf_title'
    else:  # perf_attr
        x_path_to_sents_dict = get_cached_dict('perf_sents', x_category, search_query)
        y_path_to_sents_dict, y_path_to_ids_dict = get_cached_dict(
            'attr_sents', y_category, search_query, return_both=True
        )
        title_key = 'perf_attr_title'

    matrix, sentiment_matrix, review_matrix = create_correlation_matrix(
        x_path_to_sents_dict, 
        y_path_to_ids_dict,
        y_path_to_sents_dict
    )
    
    # Ensure we don't exceed the available features
    top_n_x = min(top_n_x, matrix.shape[1] - 1)  # -1 to account for N/A
    top_n_y = min(top_n_y, matrix.shape[0] - 1)  # -1 to account for N/A
    
    # Get sums excluding N/A row/column
    x_sums = np.sum(matrix[:-1, :], axis=0)  # Exclude N/A row
    y_sums = np.sum(matrix[:, :-1], axis=1)  # Exclude N/A column
    
    # identify top x path indices from summing up the matrix (excluding N/A)
    non_zero_x = np.where(x_sums[:-1] > 0)[0]  # Exclude N/A from non-zero check
    if len(non_zero_x) == 0:  # Handle case with no non-zero values
        # If x_category is not 'all', use the selected category
        if x_category != 'all':
            # Get the index of the selected category
            x_paths = list(x_path_to_sents_dict.keys())
            try:
                selected_idx = x_paths.index(x_category)
                top_x_indices = np.array([selected_idx])
            except ValueError:
                # If category not found, use first available index
                top_x_indices = np.array([0])
        else:
            # If no category selected, use first few indices
            top_x_indices = np.arange(min(top_n_x, matrix.shape[1]-1))  # -1 to exclude N/A
    else:
        top_x_indices = non_zero_x[np.argsort(x_sums[non_zero_x])[-min(top_n_x, len(non_zero_x)):]]
    
    # Add N/A index
    top_x_indices = np.append(top_x_indices, matrix.shape[1]-1)
    
    # Get paths excluding N/A
    x_paths = list(x_path_to_sents_dict.keys())
    top_x_paths = [x_paths[i] for i in top_x_indices[:-1]]  # Exclude N/A index
    top_x_paths.append('N/A')  # Add N/A path
    
    # identify top y path indices with non-zero sum (excluding N/A)
    non_zero_y = np.where(y_sums[:-1] > 0)[0]  # Exclude N/A from non-zero check
    if len(non_zero_y) == 0:  # Handle case with no non-zero values
        # If y_category is not 'all', use the selected category
        if y_category != 'all':
            # Get the index of the selected category
            y_paths = list(y_path_to_ids_dict.keys())
            try:
                selected_idx = y_paths.index(y_category)
                top_y_indices = np.array([selected_idx])
            except ValueError:
                # If category not found, use first available index
                top_y_indices = np.array([0])
        else:
            # If no category selected, use first few indices
            top_y_indices = np.arange(min(top_n_y, matrix.shape[0]-1))  # -1 to exclude N/A
    else:
        top_y_indices = non_zero_y[np.argsort(y_sums[non_zero_y])[-min(top_n_y, len(non_zero_y)):]]
    # Add N/A index
    top_y_indices = np.append(top_y_indices, matrix.shape[0]-1)
    
    # Get paths excluding N/A
    y_paths = list(y_path_to_ids_dict.keys())
    top_y_paths = [y_paths[i] for i in top_y_indices[:-1]]  # Exclude N/A index
    top_y_paths.append('N/A')  # Add N/A path
    
    # Extract relevant submatrices
    matrix = matrix[top_y_indices, :][:, top_x_indices]
    sentiment_matrix = sentiment_matrix[top_y_indices, :][:, top_x_indices]
    review_matrix = review_matrix[top_y_indices, :][:, top_x_indices]

    # Create clean text without percentages for lookups
    if x_category == 'all':
        x_text = top_x_paths
    else:
        x_text = [path[len(x_category)+1:] if path != 'N/A' else path for path in top_x_paths]
    if y_category == 'all':
        y_text = top_y_paths
    else:
        y_text = [path[len(y_category)+1:] if path != 'N/A' else path for path in top_y_paths]
        
    # Add N/A translations to x_text and y_text where applicable
    na_translation = {
        'en': {
            'use': 'N/A (Uses with no co-mentions)',
            'perf': 'N/A (Performance aspects with no co-mentions)',
            'attr': 'N/A (Attributes with no co-mentions)',
            'attr_perf': 'N/A (Attributes/Performance with no co-mentions)'
        },
        'zh': {
            'use': 'N/A (无共同提及用途)',
            'perf': 'N/A (无共同提及性能)',
            'attr': 'N/A (无共同提及属性)',
            'attr_perf': 'N/A (无共同提及属性/性能)'
        }
    }
    
    # Determine which N/A translation to use based on plot type and axis
    if plot_type == 'use_attr_perf':
        x_na = na_translation[language]['use']
        y_na = na_translation[language]['attr_perf']
    else:  # perf_attr
        x_na = na_translation[language]['perf']
        y_na = na_translation[language]['attr']
    
    x_text = [x_na if txt == 'N/A' else txt for txt in x_text]
    y_text = [y_na if txt == 'N/A' else txt for txt in y_text]
    
    # Calculate percentages (excluding N/A from percentage calculation)
    non_na_matrix = matrix[:-1, :-1]  # Exclude N/A row and column
    total_mentions = np.sum(non_na_matrix)
    
    x_percentages = np.zeros(len(x_text))
    x_percentages[:-1] = np.sum(non_na_matrix, axis=0) / total_mentions * 100 if total_mentions > 0 else np.zeros(len(x_text)-1)
    
    y_percentages = np.zeros(len(y_text))
    y_percentages[:-1] = np.sum(non_na_matrix, axis=1) / total_mentions * 100 if total_mentions > 0 else np.zeros(len(y_text)-1)
    
    
    plot_data_cache[cache_key] = (matrix, sentiment_matrix, review_matrix, x_text, y_text, title_key, x_percentages, y_percentages)
    return plot_data_cache[cache_key]

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
     Output('y-features-slider', 'max'),
     Output('reviews-content', 'children')],
    [Input('plot-type', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('x-features-slider', 'value'),
     Input('y-features-slider', 'value'),
     Input('language-selector', 'value'),
     Input('search-button', 'n_clicks')],
    [State('search-input', 'value')]
)
def update_graph(plot_type, x_value, y_value, top_n_x, top_n_y, language, n_clicks, search_query):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None

    # Reset selections if search button was clicked
    if trigger_id == 'search-button.n_clicks':
        x_value = 'all'
        y_value = 'all'
    
    # Handle None or empty search query
    search_query = search_query if search_query else ''
    
    # Convert slider values to integers
    top_n_x = int(top_n_x)
    top_n_y = int(top_n_y)
    
    # Get data for the graph
    if plot_type == 'use_attr_perf':
        x_dict = get_cached_dict('use_sents', x_value, search_query)
        y_dict = get_cached_dict('attr_perf_ids', y_value, search_query)
    else:
        x_dict = get_cached_dict('perf_sents', x_value, search_query)
        y_dict = get_cached_dict('attr_ids', y_value, search_query)
    
    # Update slider maximums based on available features
    max_x = len(x_dict)
    max_y = len(y_dict)
    
    # Ensure top_n values don't exceed available features
    top_n_x = min(top_n_x, max_x)
    top_n_y = min(top_n_y, max_y)
    
    matrix, sentiment_matrix, review_matrix, x_text, y_text, title_key, x_percentages, y_percentages = get_plot_data(
        plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
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
        # Create display text with percentages
    x_display = [f"({perc:.1f}%) {txt}" if not txt.startswith('N/A') else txt 
             for txt, perc in zip(x_text, x_percentages)]
    y_display = [f"{txt} ({perc:.1f}%)" if not txt.startswith('N/A') else txt 
             for txt, perc in zip(y_text, y_percentages)]
    # Add the satisfaction ratio heatmap (color legend)
    fig.add_trace(go.Heatmap(
        z=sentiment_matrix,
        x=x_display,
        y=y_display,
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
            x=1.11,
            y=0.4
        )
    ))

    # Add width legend translations
    width_legend_translations = {
        'en': {
            'explanation': "Width is proportional to log(#reviews)"
        },
        'zh': {
            'explanation': "宽度与log(评论数量)成正比"
        }
    }

    # Add the custom shapes for the actual visualization
    max_mentions = np.max(matrix)
    min_mentions = np.min(matrix[matrix > 0])  # Get minimum non-zero value
    
    # Use log scale for width calculation
    def get_log_width(value, max_val, min_val):
        if value == 0:
            return 0
        # Add 1 to avoid log(1) = 0
        log_val = np.log(value + 2)
        log_max = np.log(max_val + 1)
        log_min = np.log(min_val + 1)
        # Normalize to 0.8 max width
        return 0.8 * (log_val - log_min) / (log_max - log_min + 0.000001)

    for i in range(len(y_display)):
        for j in range(len(x_display)):
            if matrix[i, j] > 0:
                box_width = get_log_width(matrix[i, j], max_mentions, min_mentions)
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
    EXAMPLE_BOX_LEFT = 1.05
    EXAMPLE_BOX_RIGHT = EXAMPLE_BOX_LEFT + 0.12
    EXAMPLE_BOX_BOTTOM = 1 
    EXAMPLE_BOX_TOP = EXAMPLE_BOX_BOTTOM + 0.11

    BRACKET_TOP = EXAMPLE_BOX_TOP + 0.06
    BRACKET_VERTICAL_LENGTH = 0.03  # Length of vertical bracket lines

    EXPLANATION_X = EXAMPLE_BOX_LEFT + (EXAMPLE_BOX_RIGHT - EXAMPLE_BOX_LEFT) * 1.5 # Center explanation above bracket
    EXPLANATION_Y = BRACKET_TOP + 0.1  # Position explanation above bracket

    # Add explanation text with translation above the bracket
    fig.add_annotation(
        xref="paper", yref="paper", 
        x=EXPLANATION_X,
        y=EXPLANATION_Y,
        text=width_legend_translations[language]['explanation'].format(example_count),
        showarrow=False,
        font=dict(size=12),
        align="right",
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

    # Update axis titles with category types
    axis_category_names = {
        'en': {
            'use': 'Uses',
            'perf': 'Performance',
            'attr': 'Attributes',
            'attr_perf': 'Attributes/Performance'
        },
        'zh': {
            'use': '用途',
            'perf': '性能',
            'attr': '属性',
            'attr_perf': '属性/性能'
        }
    }

    if plot_type == 'use_attr_perf':
        x_category_type = axis_category_names[language]['use']
        y_category_type = axis_category_names[language]['attr_perf']
    else:  # perf_attr
        x_category_type = axis_category_names[language]['perf']
        y_category_type = axis_category_names[language]['attr']

    fig.update_layout(
        title=TRANSLATIONS[language][title_key],
        xaxis=dict(
            tickangle=20,
            tickmode='array',
            tickvals=list(range(len(x_display))),
            ticktext=x_display,
            tickfont=dict(size=10),
            title=dict(
                text=f"{TRANSLATIONS[language]['x_axis_percentage']} {x_category_type}",
                font=dict(size=12)
            )
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(y_display))),
            ticktext=y_display,
            tickfont=dict(size=10),
            title=dict(
                text=f"{y_category_type} {TRANSLATIONS[language]['y_axis_percentage']}",
                font=dict(size=12)
            )
        ),
        width=dynamic_width,
        height=dynamic_height,
        margin=dict(
            l=left_margin,
            r=400,  # Increased from 300 to 400 to accommodate the explanation text
            t=150,
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

    x_options = get_options(x_value, x_text[:-1])  # Use clean text for options, excluding N/A
    y_options = get_options(y_value, y_text[-2::-1])  # Use clean text for options, excluding N/A, reverse order
    
    # Return empty list for reviews-content when search button is clicked
    reviews_content = [] if trigger_id == 'search-button.n_clicks' else dash.no_update
    
    return x_options, y_options, x_value, y_value, fig, max_x, max_y, reviews_content

@app.callback(
    Output('reviews-content', 'children', allow_duplicate=True),
    [Input('correlation-matrix', 'clickData'),
     Input('language-selector', 'value'),
     Input('search-button', 'n_clicks')],  # Add search-button as Input
    [State('plot-type', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('x-features-slider', 'value'),
     State('y-features-slider', 'value'),
     State('search-input', 'value')],
    prevent_initial_call=True
)
def display_clicked_reviews(click_data, language, n_clicks, plot_type, x_value, y_value, top_n_x, top_n_y, search_query):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    # If search button was clicked, return empty list
    if trigger_id == 'search-button.n_clicks':
        return []
        
    # Only process click data if the callback was triggered by matrix click
    if trigger_id == 'correlation-matrix.clickData' and click_data:
        # Handle None or empty search query
        search_query = search_query if search_query else ''
        
        matrix, sentiment_matrix, review_matrix, x_text, y_text, title, x_percentages, y_percentages = get_plot_data(
            plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
        )
        point = click_data['points'][0]
        
        # Get the clicked point coordinates
        clicked_x = point['x']
        clicked_y = point['y']
        
        # Create display text with percentages
        x_display = [f"({perc:.1f}%) {txt}" if not txt.startswith('N/A') else txt 
                    for txt, perc in zip(x_text, x_percentages)]
        y_display = [f"{txt} ({perc:.1f}%)" if not txt.startswith('N/A') else txt 
                    for txt, perc in zip(y_text, y_percentages)]
        
        # Find the index by matching the display text exactly
        i = y_display.index(clicked_y)
        j = x_display.index(clicked_x)
        
        if i != -1 and j != -1:
            reviews = review_matrix[i][j]
            selected_x = x_display[j]
            selected_y = y_display[i]
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
            
    return dash.no_update


def filter_reviews_by_query(reviews: List[str], query: str) -> List[str]:
    """Filter reviews based on search query using regex patterns"""
    if not query or not query.strip():
        return reviews
    
    try:
        # Convert operators and clean up query
        query = query.lower().strip()
        query = re.sub(r'\s*&\s*', ' and ', query)
        query = re.sub(r'\s*\|\s*', ' or ', query)
        
        # Convert terms to regex patterns
        terms = []
        for term in re.findall(r'\w+|[()&|]', query):
            if term in ('and', 'or', '(', ')'):
                terms.append(term)
            else:
                terms.append(f"('{term}' in review)")
        
        # Create and evaluate the filter expression
        filter_expr = ' '.join(terms)
        return [review for review in reviews if eval(filter_expr, {'re': re, 'review': review.lower()})]
    except Exception as e:
        print(f"Search query error: {str(e)}")
        return reviews

# Add new helper function to filter dictionaries
def filter_dict_by_query(path_to_sents_dict: Dict, search_query: str) -> Dict:
    """Filter path_to_sents_dict by removing reviews that don't match the query"""
    if not search_query or not search_query.strip():
        return path_to_sents_dict
        
    filtered_dict = {}
    for path, sents_dict in path_to_sents_dict.items():
        # Handle case where value is a list (ids_dict case)
        if isinstance(sents_dict, list):
            # For ids_dict, we keep the path if any associated reviews match
            # We'll need to look up the reviews in the corresponding sents_dict
            filtered_dict[path] = sents_dict
            continue
            
        # Handle case where value is a dict (sents_dict case)
        filtered_sents = {}
        for sent, val in sents_dict.items():
            if isinstance(val, dict):
                rid_reviews_dict = val
                filtered_rids = {}
                for rid, reviews in rid_reviews_dict.items():
                    matching_reviews = filter_reviews_by_query(reviews, search_query)
                    if matching_reviews:  # Only keep if there are matching reviews
                        filtered_rids[rid] = matching_reviews
                if filtered_rids:  # Only keep sentiment if there are matching reviews
                    filtered_sents[sent] = filtered_rids
            elif isinstance(val, list):
                reviews = val
                matching_reviews = filter_reviews_by_query(reviews, search_query)
                if matching_reviews:
                    filtered_sents[sent] = matching_reviews

        if filtered_sents:  # Only keep path if there are matching reviews
            filtered_dict[path] = filtered_sents
            
    return filtered_dict

# Add new callback for search examples
@app.callback(
    Output('search-examples', 'children'),
    [Input('language-selector', 'value')]
)
def update_search_examples(language):
    examples = {
        'en': [
            "Search syntax: Use & (AND), | (OR), () for grouping. Case insensitive.",
            "Examples:",
            "- quality & (durable|strong)",
            "- (good|great|excellent) & (price|cost)",
            "- (easy|simple) & install & (quick|fast)",
            "- problem & (not|never|no) & work"
        ],
        'zh': [
            "搜索语法：使用 & (且)，| (或)，() 用于分组。不区分大小写。",
            "示例：",
            "- 质量 & (耐用|坚固)",
            "- (好|优秀|完美) & (价格|成本)",
            "- (容易|简单) & 安装 & (快速|迅速)",
            "- 问题 & (不|没|无) & 工作"
        ]
    }
    return [
        html.P(
            line, 
            style={
                'marginBottom': '4px',
                'fontStyle': 'italic' if 'syntax' in line.lower() else 'normal'
            }
        ) for line in examples[language]
    ]

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')