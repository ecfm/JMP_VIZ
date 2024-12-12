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
RESULT_DIR = os.path.join(ROOT_DIR, "result/Cables_495310_v2")

color_mapping = {
        '+': '#E6F3FF',  # Blue background
        '-': '#FFE6E6',  # Red background
        '?': '#FFFFFF',  # White background
        'x': '#805AD5',  # Purple border
        'y': '#38A169',  # Green border
    }

x_highlight_en = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">Text related to X-axis</span>'
y_highlight_en = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">Text related to Y-axis</span>'
pos_highlight_en = f'<span style="background-color: {color_mapping["+"]}; color: black; padding: 5px;">Blue background indicates positive sentiment</span>'
neg_highlight_en = f'<span style="background-color: {color_mapping["-"]}; color: black; padding: 5px;">Red background indicates negative sentiment</span>'
x_highlight_zh = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">X轴相关文本</span>'
y_highlight_zh = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">Y轴相关文本</span>'
pos_highlight_zh = f'<span style="background-color: {color_mapping["+"]}; color: black; padding: 5px;">蓝底表示满意</span>'
neg_highlight_zh = f'<span style="background-color: {color_mapping["-"]}; color: black; padding: 5px;">红底表示不满意</span>'

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
        'review_format': f'Display Format: Review x: {x_highlight_en} {y_highlight_en} {pos_highlight_en} {neg_highlight_en} ASIN (Star Rating) <b>Review Title</b> Review Text',
        'percentage_explanation': 'Percentage of total mentions',
        'x_axis_percentage': 'Percentage of total mentions',
        'y_axis_percentage': 'Percentage of total mentions',
        'search_placeholder': 'e.g. (good|great)&quality or quality&(durable|strong)',
        'sentiment_filter': 'Filter Reviews:',
        'show_all': 'Show All',
        'show_positive': 'Show Positive',
        'show_negative': 'Show Negative',
        'positive_count': 'Positive Reviews: {}',
        'negative_count': 'Negative Reviews: {}',
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
        'review_format': f'显示格式： 评论 x: {x_highlight_zh} {y_highlight_zh} {pos_highlight_zh} {neg_highlight_zh} ASIN (评论星级) <b>评论标题</b> 评论内容',
        'percentage_explanation': '占总提及次数的百分比',
        'x_axis_percentage': '(该类别被提及次数%)',
        'y_axis_percentage': '(该类别被提及次数%)',
        'search_placeholder': '搜索asin或者评论关键词',
        'sentiment_filter': '评论筛选：',
        'show_all': '显示全部',
        'show_positive': '显示正面评论',
        'show_negative': '显示负面评论',
        'positive_count': '正面评论数: {}',
        'negative_count': '负面评论数: {}',
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

def extract_review_highlight(review):
    highlight, review_text = review.split('|||')
    if '<<<' in highlight:
        highlight_detail, highlight_reason = highlight.split('<<<')
        return review_text, highlight_detail, highlight_reason
    else:
        return review_text, highlight, None

def reviews_to_htmls(review_to_highlight_dict, detail_axis='x', display_reason=True):
    review_htmls = []
    pos_count = 0
    for review_text, highlights in review_to_highlight_dict.items():
        highlight_htmls = []
        pos_count_list = []
        detail_to_sent = {}
        reason_to_sent = {}
        for sent, highlight_detail, highlight_reason in highlights:
            if sent == '+':
                pos_count_list.append(1)
            elif sent == '-':
                pos_count_list.append(0)
            else:
                # print(f"Warning: {sent} is not a valid sentiment for {highlight_detail}|||{review_text}")
                pos_count_list.append(0.5)
                sent = '?'
            detail_to_sent[highlight_detail] = sent
            if highlight_reason and display_reason:
                reason_to_sent[highlight_reason] = sent
        highlight_htmls.extend([f'<span style="background-color: {color_mapping[sent]}; border: 4px solid {color_mapping[detail_axis]}; color: black; padding: 2px;">{highlight_detail}</span>' for highlight_detail in detail_to_sent.keys()])
        if display_reason:
            highlight_htmls.extend([f'<span style="background-color: {color_mapping[sent]}; border: 4px solid {color_mapping["y"]}; color: black; padding: 2px;">{highlight_reason}</span>' for highlight_reason in reason_to_sent.keys()])
        review_htmls.append(f"{' '.join(highlight_htmls)} {review_text}")
        pos_count += sum(pos_count_list)/len(pos_count_list)
    return review_htmls, pos_count

def create_correlation_matrix(x_path_to_sents_dict, y_path_to_ids_dict, y_path_to_sents_dict):
    # Add N/A entries
    
    matrix = np.zeros((len(y_path_to_ids_dict)+1, len(x_path_to_sents_dict)+1)) # +1 for N/A
    sentiment_matrix = np.zeros((len(y_path_to_ids_dict)+1, len(x_path_to_sents_dict)+1)) # +1 for N/A
    review_matrix = np.empty((len(y_path_to_ids_dict)+1, len(x_path_to_sents_dict)+1), dtype=object) # +1 for N/A
    
    # Initialize each cell with its own empty list
    for i in range(review_matrix.shape[0]):
        for j in range(review_matrix.shape[1]):
            review_matrix[i, j] = []
    
    x_no_match_review_to_highlight_dict = [defaultdict(list) for _ in range(len(x_path_to_sents_dict))]
    for j, (path2, x_sents_dict) in enumerate(x_path_to_sents_dict.items()):
        for sent, reason_rids in x_sents_dict.items():
            for rid, reviews in reason_rids.items():
                for review in reviews:
                    review_text, highlight_detail, highlight_reason = extract_review_highlight(review)
                    x_no_match_review_to_highlight_dict[j][review_text].append((sent, highlight_detail, highlight_reason))
    y_no_match_review_to_highlight_dict = [defaultdict(list) for _ in range(len(y_path_to_ids_dict))]

    # Process matrix including N/A
    for i, ((path1, y_ids), y_sents_dict) in enumerate(zip(y_path_to_ids_dict.items(), y_path_to_sents_dict.values())):
        for sent, reviews in y_sents_dict.items():
            for review in reviews:
                review_text, highlight_detail, highlight_reason = extract_review_highlight(review)
                y_no_match_review_to_highlight_dict[i][review_text].append((sent, highlight_detail, highlight_reason))
        for j, (path2, x_sents_dict) in enumerate(x_path_to_sents_dict.items()):
            x_sent_ids = set()
            review_to_highlight_dict = defaultdict(list)
            
            # First collect all reviews and sentiments for this x category
            for sent, reason_rids in x_sents_dict.items():
                x_sent_ids.update(reason_rids.keys())
                for rid, reviews in reason_rids.items():
                    if rid in y_ids:  # Co-mention case
                        for review in reviews:
                            review_text, highlight_detail, highlight_reason = extract_review_highlight(review)
                            if review_text in y_no_match_review_to_highlight_dict[i]:
                                del y_no_match_review_to_highlight_dict[i][review_text]
                            if review_text in x_no_match_review_to_highlight_dict[j]:
                                del x_no_match_review_to_highlight_dict[j][review_text]
                            review_to_highlight_dict[review_text].append((sent, highlight_detail, highlight_reason))
            review_with_highlight, pos_count = reviews_to_htmls(review_to_highlight_dict, detail_axis='x', display_reason=True)
            review_matrix[i, j] = review_with_highlight
            sentiment_matrix[i, j] = pos_count
            # Update counts and sentiment
            matrix[i, j] = len(review_with_highlight)
            if matrix[i, j] > 0:
                sentiment_matrix[i, j] = pos_count / matrix[i, j]
    for i in range(len(y_no_match_review_to_highlight_dict)):
        matrix[i, -1] = len(y_no_match_review_to_highlight_dict[i])
        review_with_highlight, pos_count = reviews_to_htmls(y_no_match_review_to_highlight_dict[i], detail_axis='y', display_reason=False)
        review_matrix[i, -1] = review_with_highlight
        if matrix[i, -1] > 0:
            sentiment_matrix[i, -1] = pos_count / matrix[i, -1]
    for j in range(len(x_no_match_review_to_highlight_dict)):
        matrix[-1, j] = len(x_no_match_review_to_highlight_dict[j])
        review_with_highlight, pos_count = reviews_to_htmls(x_no_match_review_to_highlight_dict[j], detail_axis='x', display_reason=False)
        review_matrix[-1, j] = review_with_highlight
        if matrix[-1, j] > 0:   
            sentiment_matrix[-1, j] = pos_count / matrix[-1, j]
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
])

# Remove the search_box definition from the global scope and move it into a function
def create_search_box(language='en'):  # Default to English
    return html.Div([
        html.Div([
            dcc.Input(
                id='search-input',
                type='text',
                placeholder=TRANSLATIONS[language]['search_placeholder'],
                n_submit=0,  # Add this for Enter key handling
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
        html.Div(id='search-results-info', style={
            'marginTop': '10px',
            'fontSize': '14px',
            'color': '#444444',  # Darker grey text
            'display': 'inline-block',  # Make div only as wide as content
            'backgroundColor': '#FFEB3B',  # Bright yellow background
            'padding': '0 4px',  # Add small horizontal padding
        }),
        html.Div([
            html.Div(id='search-examples', style={
                'marginTop': '8px', 
                'fontSize': '12px',
                'color': '#666'
            })
        ])
    ], style={'marginBottom': '20px'})

# Update main_layout to use the function
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
        create_search_box(),  # Use the function here
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
            # Initialize RadioItems with horizontal layout
            dcc.RadioItems(
                id='sentiment-filter',
                options=[
                    {'label': 'Show All', 'value': 'show_all'},
                    {'label': 'Show Positive', 'value': 'show_positive'},
                    {'label': 'Show Negative', 'value': 'show_negative'}
                ],
                value='show_all',
                style={'display': 'none'},  # Initially hidden
                className='sentiment-filter-horizontal'  # Add class for styling
            ),
            html.Div(id='review-counts', style={'margin': '5px', 'fontSize': '1em', 'color': '#444'}),
            html.Div(id='reviews-content', style={'maxHeight': '300px', 'overflowY': 'auto', 'border': '2px solid #ddd', 'borderRadius': '5px', 'padding': '10px'})
        ], style={'borderTop': '3px double #ddd'})
    ])
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    # Move language selector to upper left and adjust styling
    html.Div([
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
    
    def process_axis(matrix: np.ndarray, axis: int, top_n: int, paths: list) -> tuple:
        """Process one axis (x or y) to get indices, paths and percentages.
        Args:
            matrix: The correlation matrix
            axis: 0 for y-axis (rows), 1 for x-axis (columns)
            top_n: Number of top features to return
            paths: List of paths for this axis
        Returns:
            tuple of (top_indices, top_paths, percentages)
        """
        sums = np.sum(matrix, axis=1-axis)  # sum over opposite axis
        non_zero = np.where(sums > 0)[0]
        if len(non_zero) == 0:
            # If there are no non-zero values, show the first top_n features
            top_indices = np.arange(min(top_n, len(sums)))
        else:
            top_indices = non_zero[np.argsort(sums[non_zero])[-min(top_n, len(non_zero)):]]
        top_paths = [paths[i] for i in top_indices]
        total_mentions = np.sum(matrix)
        percentages = np.sum(matrix[top_indices] if axis == 0 else matrix[:, top_indices], axis=1-axis) / total_mentions * 100 if total_mentions > 0 else np.zeros(len(top_indices))
        return top_indices, top_paths, percentages
    
    # Process x and y axes
    x_paths = list(x_path_to_sents_dict.keys()) + ['N/A']
    y_paths = list(y_path_to_ids_dict.keys()) + ['N/A']
    top_x_indices, top_x_paths, x_percentages = process_axis(matrix, 1, top_n_x, x_paths)
    top_y_indices, top_y_paths, y_percentages = process_axis(matrix, 0, top_n_y, y_paths)
    
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
    
    x_display = [f"({perc:.1f}%) {txt}" for txt, perc in zip(x_text, x_percentages)]
    y_display = [f"{txt} ({perc:.1f}%)" for txt, perc in zip(y_text, y_percentages)]
    
    plot_data_cache[cache_key] = (matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key)
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
     Output('reviews-content', 'children'),
     Output('x-features-slider', 'value'),  # Add new output for x slider value
     Output('y-features-slider', 'value')], # Add new output for y slider value
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
    
    if x_value is None:
        x_value = 'all'
    if y_value is None:
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
    
    # For search reset, set slider values to min(7, max available)
    if trigger_id == 'search-button.n_clicks':
        top_n_x = min(7, max_x)  # Reset to default of 7 or max available
        top_n_y = min(7, max_y)  # Reset to default of 7 or max available
    else:
        # Ensure top_n values don't exceed available features
        top_n_x = min(top_n_x, max_x)
        top_n_y = min(top_n_y, max_y)
    
    matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key = get_plot_data(
        plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
    )

    
    # Calculate dynamic dimensions based on number of features
    # Height calculation
    base_height = 400
    min_height = 300
    additional_height_per_feature = 40
    dynamic_height = max(
        min_height,
        base_height + max(0, len(y_display) - 5) * additional_height_per_feature
    )
    
    # Width calculation - increase base width to accommodate legends
    base_width = 800  
    min_width = 1200 
    additional_width_per_feature = 60
    max_width = 2000
    
    # Calculate width based on the length of x-axis labels and number of features
    avg_label_length = sum(len(str(label)) for label in x_display) / len(x_display) if x_display else 0
    width_factor = max(1, avg_label_length / 15)  # Adjust width more for longer labels
    
    dynamic_width = min(
        max_width,
        max(
            min_width,
            base_width + max(0, len(x_display) - 5) * additional_width_per_feature * width_factor
        )
    )
    
    # Calculate dynamic margins based on label lengths
    max_y_label_length = max(len(str(label)) for label in y_display) if y_display else 0
    left_margin = min(300, max(150, max_y_label_length * 8))  # Increased margins
    right_margin = 450  # Increased from 400

    # Create the base heatmap figure
    fig = go.Figure()

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
            r=right_margin,  # Using the new right_margin
            t=150,
            b=100,
            autoexpand=True
        )
    )

    # First, add the needed translations to both language dictionaries
    hover_translations = {
        'en': {
            'hover_x': 'X-axis',
            'hover_y': 'Y-axis',
            'hover_count': 'Count',
            'hover_satisfaction': 'Satisfaction Ratio'
        },
        'zh': {
            'hover_x': 'X轴',
            'hover_y': 'Y轴',
            'hover_count': '数量',
            'hover_satisfaction': '满意度比例'
        }
    }

    # Update hover template with translations
    hover_template = (
        f"{hover_translations[language]['hover_x']}: %{{x}}<br>" +
        f"{hover_translations[language]['hover_y']}: %{{y}}<br>" +
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
    
    # Return both the new slider values and the rest of the outputs
    return (x_options, y_options, x_value, y_value, fig, max_x, max_y, 
            reviews_content, top_n_x, top_n_y)

@app.callback(
    [Output('reviews-content', 'children', allow_duplicate=True),
     Output('sentiment-filter', 'value'),
     Output('sentiment-filter', 'style'),
     Output('review-counts', 'children')],
    [Input('sentiment-filter', 'value'),
     Input('correlation-matrix', 'clickData'),
     Input('language-selector', 'value')],
    [State('plot-type', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value'),
     State('x-features-slider', 'value'),
     State('y-features-slider', 'value'),
     State('search-input', 'value')],
    prevent_initial_call=True
)
def update_reviews_with_sentiment_filter(sentiment_filter, click_data, language, plot_type, x_value, y_value, top_n_x, top_n_y, search_query):
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'] if ctx.triggered else None
    
    if trigger_id == 'search-button.n_clicks':
        return [], 'show_all', {'display': 'none'}, ''
        
    if trigger_id == 'correlation-matrix.clickData':
        sentiment_filter = 'show_all'
    
    if click_data:
        point = click_data['points'][0]
        clicked_x = point['x']
        clicked_y = point['y']
        
        search_query = search_query if search_query else ''
        
        matrix, sentiment_matrix, review_matrix, x_text, x_display, y_text, y_display, title_key = get_plot_data(
            plot_type, x_value, y_value, top_n_x, top_n_y, language, search_query
        )
        
        try:
            i = y_display.index(clicked_y)
            j = x_display.index(clicked_x)
            
            if i != -1 and j != -1:
                reviews = review_matrix[i][j]
                
                # Count positive and negative reviews - moved outside the sentiment filter condition
                positive_reviews = []
                negative_reviews = []
                neutral_reviews = []
                for review in reviews:
                    if f'<span style="background-color: {color_mapping["+"]}' in review:
                        positive_reviews.append(review)
                    elif f'<span style="background-color: {color_mapping["-"]}' in review:
                        negative_reviews.append(review)
                    else:
                        neutral_reviews.append(review)
                
                # Filter reviews based on selection
                if sentiment_filter == 'show_positive':
                    filtered_reviews = positive_reviews
                elif sentiment_filter == 'show_negative':
                    filtered_reviews = negative_reviews
                else:  # 'all'
                    filtered_reviews = reviews
                
                x_clicked = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["x"]}; color: black; padding: 5px;">X=<b>{clicked_x}</b></span>'
                y_clicked = f'<span style="background-color: {color_mapping["?"]}; border: 5px solid {color_mapping["y"]}; color: black; padding: 5px;">Y=<b>{clicked_y}</b></span>'
                
                # Create header HTML with sentiment filter
                header_html = f"""
                <html>
                <head>
                    <style>
                        body {{
                            margin: 10px;
                            font-family: sans-serif;
                            line-height: 1.5;
                            padding: 2px;
                            border-bottom: 2px ridge #eee;
                        }}
                        span {{
                            display: inline-block;
                            margin: 2px;
                        }}
                    </style>
                </head>
                <body>
                    <div>{TRANSLATIONS[language]['selected']} {x_clicked}, {y_clicked}</div>
                    <div style="margin-top: 5px;">
                        {TRANSLATIONS[language]['review_format']}
                    </div>
                </body>
                </html>
                """

                content = [
                    html.Iframe(
                        srcDoc=header_html,
                        style={
                            'width': '100%',
                            'height': '150px',
                            'border': 'none',
                            'backgroundColor': 'white'
                        }
                    )
                ]
                
                if filtered_reviews:
                    reviews_html = []
                    for idx, review in enumerate(filtered_reviews):
                        reviews_html.append(f"""
                            <div class="review-container">
                                <span>{TRANSLATIONS[language]['review']} {idx + 1}: </span>
                                {review}
                            </div>
                        """)
                    
                    html_content = f"""
                    <html>
                    <head>
                        <style>
                            body {{
                                margin: 0;
                                font-family: sans-serif;
                                line-height: 1.5;
                            }}
                            .review-container {{
                                margin: 0;
                                padding: 5px;
                                border-bottom: 1px solid #eee;
                            }}
                            .review-container:last-child {{
                                border-bottom: none;
                            }}
                        </style>
                    </head>
                    <body>
                        {''.join(reviews_html)}
                    </body>
                    </html>
                    """
                    
                    content.append(html.Iframe(
                        srcDoc=html_content,
                        style={
                            'width': '100%',
                            'height': '300px',
                            'border': 'none',
                            'borderRadius': '5px',
                            'backgroundColor': 'white'
                        }
                    ))
                else:
                    content.append(html.P(TRANSLATIONS[language]['no_reviews']))
                
                # Create count display based on filter selection
                count_display = []
                if sentiment_filter == 'show_all':
                    count_display = [
                        html.Span(f"{TRANSLATIONS[language]['positive_count'].format(len(positive_reviews))} | "),
                        html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))
                    ]
                elif sentiment_filter == 'show_positive':
                    count_display = [html.Span(TRANSLATIONS[language]['positive_count'].format(len(positive_reviews)))]
                else:  # show_negative
                    count_display = [html.Span(TRANSLATIONS[language]['negative_count'].format(len(negative_reviews)))]
                
                # Update filter style to show horizontally
                filter_style = {
                    'display': 'block',
                    'marginBottom': '10px',
                    'whiteSpace': 'nowrap',  # Prevent wrapping
                    'display': 'flex',
                    'flexDirection': 'row',
                    'gap': '20px',  # Space between options
                    'alignItems': 'center'
                }
                
                return content, sentiment_filter, filter_style, count_display
                
        except ValueError as e:
            print(f"Error processing click data: {str(e)}")
            
    return dash.no_update, sentiment_filter, dash.no_update, dash.no_update

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
        ],
        'zh': [
            "搜索语法：使用 & (且)，| (或)，() 用于分组。不区分大小写。示例：",
            "- B08NJ2SGDW | B08NJ2SGDW: 搜索asin为 B08NJ2SGDW 或 B08NJ2SGDW 的评论",
            "- (B08NJ2SGDW|B08NJ2SGDW)&(good|great): 搜索asin为 B08NJ2SGDW 或 B08NJ2SGDW 并且评论中包含 good 或 great 的评论",
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

def normalize_search_query(query: str) -> str:
    """Normalize search query for display by quoting terms and using readable operators"""
    if not query or not query.strip():
        return ""
        
    query = query.strip()
    
    # First handle parentheses spacing
    query = re.sub(r'\(\s*', '(', query)
    query = re.sub(r'\s*\)', ')', query)
    
    # Split into tokens while preserving operators and parentheses
    tokens = re.findall(r'\(|\)|\w+|&|\|', query)
    
    # Process tokens
    normalized = []
    for token in tokens:
        if token in ('&', '|'):
            # Replace operators
            normalized.append('and' if token == '&' else 'or')
        elif token in ('(', ')'):
            # Keep parentheses as-is
            normalized.append(token)
        else:
            # Quote terms
            normalized.append(f'"{token}"')
    
    # Join with proper spacing
    return ' '.join(normalized)

@app.callback(
    Output('search-results-info', 'children'),
    [Input('search-button', 'n_clicks'),
     Input('language-selector', 'value')],
    [State('search-input', 'value'),
     State('plot-type', 'value'),
     State('x-axis-dropdown', 'value'),
     State('y-axis-dropdown', 'value')]
)
def update_search_results_info(n_clicks, language, search_query, plot_type, x_value, y_value):
    if not search_query or not search_query.strip():
        return ''
        
    # Get the dictionaries based on plot type
    if plot_type == 'use_attr_perf':
        x_dict = get_cached_dict('use_sents', x_value, search_query)
        y_dict = get_cached_dict('attr_perf_sents', y_value, search_query)
    else:
        x_dict = get_cached_dict('perf_sents', x_value, search_query)
        y_dict = get_cached_dict('attr_sents', y_value, search_query)
    
    # Count total unique reviews
    total_reviews = set()
    
    # Count reviews in x_dict
    for sents_dict in x_dict.values():
        for sent_dict in sents_dict.values():
            if isinstance(sent_dict, dict):
                for reviews in sent_dict.values():
                    total_reviews.update(reviews)
            elif isinstance(sent_dict, list):
                total_reviews.update(sent_dict)
                
    # Count reviews in y_dict
    for sents_dict in y_dict.values():
        for sent_dict in sents_dict.values():
            if isinstance(sent_dict, dict):
                for reviews in sent_dict.values():
                    total_reviews.update(reviews)
            elif isinstance(sent_dict, list):
                total_reviews.update(sent_dict)
    
    # Normalize search query
    normalized_query = normalize_search_query(search_query)
    
    # Return translated message with results
    translations = {
        'en': {
            'results': f"Found {len(total_reviews)} reviews matching query: {normalized_query}",
        },
        'zh': {
            'results': f"找到 {len(total_reviews)} 条匹配的评论，搜索条件：{normalized_query}",
        }
    }
    
    return translations[language]['results']

# Add callback to update placeholder when language changes
@app.callback(
    Output('search-input', 'placeholder'),
    [Input('language-selector', 'value')]
)
def update_search_placeholder(language):
    return TRANSLATIONS[language]['search_placeholder']

# Add new callback for handling Enter key press
@app.callback(
    Output('search-button', 'n_clicks'),
    [Input('search-input', 'n_submit')],
    [State('search-button', 'n_clicks')]
)
def handle_enter_press(n_submit, current_clicks):
    if n_submit:
        return (current_clicks or 0) + 1
    return dash.no_update

@app.callback(
    Output('sentiment-filter', 'options'),
    [Input('language-selector', 'value')]
)
def update_sentiment_filter_options(language):
    return [
        {'label': TRANSLATIONS[language]['show_all'], 'value': 'show_all'},
        {'label': TRANSLATIONS[language]['show_positive'], 'value': 'show_positive'},
        {'label': TRANSLATIONS[language]['show_negative'], 'value': 'show_negative'}
    ]

# Add CSS styles to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .sentiment-filter-horizontal .radio-item {
                display: inline-block;
                margin-right: 20px;
            }
            .sentiment-filter-horizontal input[type="radio"] {
                margin-right: 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')