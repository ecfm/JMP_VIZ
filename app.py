import os
import dash
from flask_login import LoginManager
from dash import html, dcc
from dash.dependencies import Input, Output, State
from urllib.parse import parse_qs

from layouts import get_app_layout
from callbacks import register_callbacks, User
from config import VALID_USERNAME, plot_data_cache, get_result_dir

# Initialize Dash app and login manager
app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# Add the secret key settings for Flask-Login
server.config.update(
    SECRET_KEY='your-secret-key-here',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True if os.environ.get('PRODUCTION', False) else False,
)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = '/login'

@login_manager.user_loader
def load_user(username):
    if username == VALID_USERNAME:
        return User(username)
    return None

# Set the app layout
app.layout = get_app_layout()

# Add CSS styles to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Correlation Matrix Visualization</title>
        {%favicon%}
        {%css%}
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

# Add callback to parse URL parameters for language and category selection
@app.callback(
    [Output('app-language-state', 'children'),
     Output('category-state', 'children')],
    [Input('url', 'search')]
)
def update_parameters_from_url(search):
    """Parse URL parameters for language and category and update state."""
    # Default values
    lang = 'zh'  # Default to Chinese
    category = 'Cables'  # Default category
    
    if search:
        # Parse the query string
        params = parse_qs(search.lstrip('?'))
        
        # Get language parameter
        if 'lang' in params:
            temp_lang = params['lang'][0]
            # Validate the language parameter
            if temp_lang in ['en', 'zh']:
                lang = temp_lang
        
        # Get category parameter
        if 'category' in params:
            category = params['category'][0]
    
    return lang, category

# Register all callbacks
register_callbacks(app)

# Clear the plot data cache to ensure no cached results with N/A entries remain
plot_data_cache.clear()

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 