from dash import html, dcc
from config import TRANSLATIONS
from utils import create_search_box


def get_login_layout(language='en'):
    """Create the login page layout."""
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


def get_main_layout(language='en'):
    """Create the main application layout."""
    
    # Create bar chart controls that will only be shown for bar chart view
    bar_chart_controls = html.Div(
        id='bar-chart-controls',
        children=[
            html.Label(id='bar-category-label'),
            html.Div([
                dcc.Checklist(
                    id='bar-category-checklist',
                    options=[
                        {'label': TRANSLATIONS[language]['usage_category'], 'value': 'usage'},
                        {'label': TRANSLATIONS[language]['attribute_category'], 'value': 'attribute'},
                        {'label': TRANSLATIONS[language]['performance_category'], 'value': 'performance'}
                    ],
                    value=['usage', 'attribute', 'performance'],
                    inline=True
                ),
            ], style={'marginBottom': '10px'}),
            html.Label(id='bar-zoom-label'),
            dcc.Dropdown(
                id='bar-zoom-dropdown',
                options=[{'label': TRANSLATIONS[language]['all_level_0'], 'value': 'all'}],
                value='all',
                placeholder="Select a category to zoom"
            ),
            html.Div([
                html.Label(id='bar-count-label', children=TRANSLATIONS[language]['num_x_features'], style={'marginTop': '10px'}),
                dcc.Slider(
                    id='bar-count-slider',
                    min=1,
                    max=50,  # Initial max that will be updated dynamically
                    value=10,
                    step=1,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], style={'marginTop': '15px'})
        ],
        style={'display': 'none'}  # Initially hidden
    )
    
    # Create matrix view controls (sliders and dropdowns)
    matrix_view_controls = html.Div([
        html.Div([
            html.Label(id='y-axis-label'),
            dcc.Dropdown(
                id='y-axis-dropdown',
                options=[{'label': TRANSLATIONS[language]['all_level_0'], 'value': 'all'}],
                value='all',
                placeholder="Select a category for Y-axis"
            ),
            html.Label(id='num-y-features-label', style={'marginTop': '10px'}),
            dcc.Slider(
                id='y-features-slider',
                min=1,
                max=10,
                value=7,
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            html.Label(id='x-axis-label'),
            dcc.Dropdown(
                id='x-axis-dropdown',
                options=[{'label': TRANSLATIONS[language]['all_level_0'], 'value': 'all'}],
                value='all',
                placeholder="Select a category for X-axis"
            ),
            html.Label(id='num-x-features-label', style={'marginTop': '10px'}),
            dcc.Slider(
                id='x-features-slider',
                min=1,
                max=10,
                value=7,
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], id='matrix-view-controls', style={'display': 'block'})  # Initially visible
    
    return html.Div([
        html.Div([
            html.Div([
                html.Button(
                    TRANSLATIONS[language]['logout'], 
                    id='logout-button', 
                    style={'float': 'right', 'margin': '20px'}
                ),
            ]),
            create_search_box(language),
            html.Div([
                html.Label(id='plot-type-label'),
                dcc.RadioItems(
                    id='plot-type',
                    options=[
                        {'label': TRANSLATIONS[language]['bar_chart'], 'value': 'bar_chart'},
                        {'label': TRANSLATIONS[language]['use_vs_attr_perf'], 'value': 'use_attr_perf'},
                        {'label': TRANSLATIONS[language]['perf_vs_attr'], 'value': 'perf_attr'}
                    ],
                    value='use_attr_perf',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style={'width': '100%', 'display': 'inline-block', 'margin-bottom': '10px'}),
            
            # Bar chart controls section
            bar_chart_controls,
            
            # Matrix view controls section (with sliders)
            matrix_view_controls,
            
        ]),
        html.Div([
            dcc.Graph(
                id='correlation-matrix', 
                style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px'}
            ),
            html.Div([
                html.H3(TRANSLATIONS[language]['reviews']),
                # Initialize RadioItems with horizontal layout
                dcc.RadioItems(
                    id='sentiment-filter',
                    options=[
                        {'label': TRANSLATIONS[language]['show_all'], 'value': 'show_all'},
                        {'label': TRANSLATIONS[language]['show_positive'], 'value': 'show_positive'},
                        {'label': TRANSLATIONS[language]['show_negative'], 'value': 'show_negative'}
                    ],
                    value='show_all',
                    style={'display': 'none'},  # Initially hidden
                    className='sentiment-filter-horizontal'  # Add class for styling
                ),
                html.Div(
                    id='review-counts', 
                    style={'margin': '5px', 'fontSize': '1em', 'color': '#444'}
                ),
                html.Div(
                    id='reviews-content', 
                    style={
                        'maxHeight': '1200px', 
                        'overflowY': 'auto', 
                        'border': '2px solid #ddd', 
                        'borderRadius': '5px', 
                        'padding': '10px'
                    }
                )
            ], style={'borderTop': '3px double #ddd'})
        ])
    ])


def get_app_layout():
    """Create the overall application layout."""
    return html.Div([
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
                    style={'float': 'right', 'margin': '10px'}
                ),
            ], style={'width': '100%', 'clear': 'both'}),
        ], style={'width': '100%', 'clear': 'both'}),
        html.Div(id='page-content')
    ]) 