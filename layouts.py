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


def get_main_layout(language='zh'):
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
                html.Div([
                    html.Button(
                        TRANSLATIONS[language]['select_all'],
                        id='select-all-button',
                        n_clicks=0,
                        style={'marginRight': '10px', 'fontSize': '0.8em'}
                    ),
                    html.Button(
                        TRANSLATIONS[language]['unselect_all'],
                        id='unselect-all-button',
                        n_clicks=0,
                        style={'fontSize': '0.8em'}
                    )
                ], style={'display': 'inline-block', 'marginLeft': '20px'})
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
                max=20,
                value=10,
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
                max=20,
                value=10,
                step=1,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], id='matrix-view-controls', style={'display': 'block'})  # Initially visible
    
    # Create date range slider
    date_filter_controls = html.Div(
        id='date-filter-controls',
        children=[
            html.Div([
                html.Label(TRANSLATIONS[language]['date_filter_label'], style={'marginBottom': '5px'}),
                # Add visible display for selected date range
                html.Div(id='date-range-display', style={
                    'marginTop': '10px',
                    'fontSize': '1.1em',
                    'color': '#333',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'backgroundColor': '#f0f0f0',
                    'padding': '8px',
                    'borderRadius': '4px'
                })
            ]),
            dcc.RangeSlider(
                id='date-filter-slider',
                min=0,
                max=1,  # This will be updated in a callback
                value=[0, 1],  # Initial values will be replaced in a callback
                marks={},  # This will be updated in a callback 
                updatemode='mouseup'  # Only trigger updates on mouseup for better performance
            ),
            # Hidden div to store actual date values as strings
            html.Div(id='date-filter-storage', style={'display': 'none'})
        ],
        style={'marginTop': '10px', 'marginBottom': '15px'}
    )
    
    return html.Div([
        html.Div([
            # Create a header area with logout and language selector
            html.Div([
                html.Div([
                    # Single visible language selector
                    dcc.RadioItems(
                        id='language-selector',
                        options=[
                            {'label': 'English', 'value': 'en'},
                            {'label': '中文', 'value': 'zh'}
                        ],
                        value=language,  # Use the current language value
                        style={'float': 'left', 'margin': '20px'}
                    ),
                    # Logout button
                    html.Button(
                        TRANSLATIONS[language]['logout'], 
                        id='logout-button', 
                        style={'float': 'right', 'margin': '20px'}
                    ),
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'width': '100%'})
            ], style={'marginBottom': '10px', 'borderBottom': '1px solid #ddd', 'paddingBottom': '5px'}),
            
            # Add a div to display total reviews count in a box with border
            html.Div([
                html.H4(id='total-reviews-count', children='', style={
                    'margin': '5px 0',
                    'padding': '10px',
                    'textAlign': 'center',
                    'fontWeight': 'bold'
                })
            ], style={
                'margin': '10px 0',
                'border': '2px solid #ddd',
                'borderRadius': '5px',
                'backgroundColor': '#f9f9f9',
                'boxShadow': '0 1px 3px rgba(0,0,0,0.1)'
            }),
            # Add date filter controls before the search box
            date_filter_controls,
            # Move the search box below the date filter
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
                    value='bar_chart',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                ),
            ], style={'width': '100%', 'display': 'inline-block', 'margin-bottom': '10px'}),
            
            # Bar chart controls section
            bar_chart_controls,
            
            # Matrix view controls section (with sliders)
            matrix_view_controls,
            
        ]),
        html.Div([
            # Add trend chart for bar chart view
            html.Div([
                html.Div([
                    html.Label(id='time-bucket-label', children='Time interval:'),
                    dcc.Dropdown(
                        id='time-bucket-dropdown',
                        options=[
                            {'label': 'Month', 'value': 'month'},
                            {'label': '3 Months', 'value': '3month'},
                            {'label': '6 Months', 'value': '6month'},
                            {'label': 'Year', 'value': 'year'}
                        ],
                        value='3month',
                        style={'width': '200px', 'display': 'inline-block'},
                        clearable=False
                    )
                ], style={'marginBottom': '10px'}),
                dcc.Graph(
                    id='trend-chart', 
                    style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'marginBottom': '20px'}
                )
            ],
            id='trend-chart-container',
            style={'display': 'none'}  # Initially hidden, shown only for bar chart
            ),
            dcc.Graph(
                id='main-figure', 
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
                # Hidden div to store selected words
                html.Div(id='selected-words-storage', children='[]', style={'display': 'none'}),
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
        # Login content - initially hidden
        html.Div(
            get_login_layout('zh'),  # Default to English
            id='login-content',
            style={'display': 'none'}
        ),
        # Main content - initially hidden
        html.Div(
            get_main_layout('zh'),  # Default to English
            id='main-content',
            style={'display': 'none'}
        )
    ]) 