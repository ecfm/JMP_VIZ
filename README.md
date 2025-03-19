# Correlation Matrix Visualization

A Dash-based web application for visualizing correlations between different categories of product attributes and performance metrics.

## Project Structure

The application is organized into these main files:

- `app.py` - Main application entry point and server configuration
- `config.py` - Constants, configurations, and translations
- `data.py` - Data loading and processing functions
- `layouts.py` - UI layouts (login and main application)
- `callbacks.py` - Callback functions for interactivity
- `utils.py` - Utility functions for plotting and other helpers
- `assets/custom.css` - Custom CSS styles

## Setup and Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Access the application in your web browser at http://localhost:8080

## Authentication

The application uses Flask-Login for authentication with the following credentials:
- Username: xsight
- Password: mit100

## Features

- Interactive correlation matrix visualization
- Bi-lingual support (English and Chinese)
- Filtering by categories and search queries
- Review display with sentiment highlighting
- Responsive design

## Data Structure

The application visualizes relationships between different categories:
- Uses vs. Attributes + Performance
- Performance vs. Attributes

Data is loaded from JSON files in the `result/Ribbons` directory. 