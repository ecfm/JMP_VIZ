import os
import dash
from flask_login import LoginManager
from dash.dependencies import Input, Output
from urllib.parse import parse_qs
import logging

from layouts import get_app_layout
from callbacks import register_callbacks, User
from config import VALID_USERNAME, plot_data_cache
from logging_config import log_app_event, log_error

# Initialize Dash app and login manager
app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True

# Log application startup
log_app_event("Application starting")

# Add the secret key settings for Flask-Login
try:
    server.config.update(
        SECRET_KEY='your-secret-key-here',
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=True if os.environ.get('PRODUCTION', False) else False,
    )
    log_app_event("Server configuration updated")
except Exception as e:
    log_error("Failed to update server configuration", exc_info=True)

# Setup Flask-Login
try:
    login_manager = LoginManager()
    login_manager.init_app(server)
    login_manager.login_view = '/login'
    log_app_event("Login manager initialized")
except Exception as e:
    log_error("Failed to initialize login manager", exc_info=True)

@login_manager.user_loader
def load_user(username):
    try:
        if username == VALID_USERNAME:
            log_app_event(f"User loaded: {username}")
            return User(username)
        log_app_event(f"Invalid user load attempt: {username}")
        return None
    except Exception as e:
        log_error(f"Error loading user: {username}", exc_info=True)
        return None

# Set the app layout
try:
    app.layout = get_app_layout()
    log_app_event("Application layout set")
except Exception as e:
    log_error("Failed to set application layout", exc_info=True)

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
    try:
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
        
        log_app_event(f"URL parameters parsed: language={lang}, category={category}")
        return lang, category
    except Exception as e:
        log_error("Error parsing URL parameters", exc_info=True)
        # Return defaults in case of error
        return 'zh', 'Cables'

# Register all callbacks
try:
    register_callbacks(app)
    log_app_event("Callbacks registered")
except Exception as e:
    log_error("Failed to register callbacks", exc_info=True)

# Clear the plot data cache to ensure no cached results with N/A entries remain
try:
    plot_data_cache.clear()
    log_app_event("Plot data cache cleared")
except Exception as e:
    log_error("Failed to clear plot data cache", exc_info=True)

# Run the app
if __name__ == '__main__':
    try:
        log_app_event("Starting application server")
        app.run(debug=True, host='0.0.0.0', port=8080)
    except Exception as e:
        log_error("Application server failed to start", exc_info=True) 