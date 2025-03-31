import dash
from dash import html, dcc
from urllib.parse import urlparse, parse_qs

def get_category_from_url():
    """Get category from URL query parameters or default to 'Cables'."""
    try:
        # Get the current URL from the browser
        current_url = dash.get_app().request.url
        parsed_url = urlparse(current_url)
        query_params = parse_qs(parsed_url.query)
        return query_params.get('category', ['Cables'])[0]
    except:
        # If we can't get the URL, return default
        return 'Cables' 