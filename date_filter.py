import json
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from data import get_review_date_range
import dash
from dash import html, dcc
from config import TRANSLATIONS
def handle_date_filter(trigger_id, date_slider_value, date_filter_storage):
    """
    Handle date filter initialization and updates.
    
    Args:
        trigger_id: ID of the component that triggered the callback
        date_slider_value: Current value of the date slider
        date_filter_storage: Current date filter storage JSON
        
    Returns:
        Tuple containing (updated_date_storage, slider_min, slider_max, slider_value, slider_marks)
    """
    # Default slider settings for no-update cases
    slider_min = dash.no_update
    slider_max = dash.no_update
    slider_value = dash.no_update
    slider_marks = dash.no_update
    
    # Initialize storage if empty or on first load
    storage_data = {}
    if not date_filter_storage:
        # Get min and max dates from all reviews
        min_date_str, max_date_str = get_review_date_range()
        
        if min_date_str is None or max_date_str is None:
            # No dates found, use empty values
            return json.dumps({"start_date": None, "end_date": None}), 0, 1, [0, 1], {}
        
        # Convert to datetime objects
        min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
        
        # Set up slider values
        slider_min = 0
        slider_max = (max_date - min_date).days
        slider_value = [slider_min, slider_max]
        
        # Create slider marks
        slider_marks = {}
        
        # Determine appropriate frequency based on date range
        date_range_days = (max_date - min_date).days
        
        # Use quarterly marks for ranges under 3 years, 6-monthly for 3-10 years, yearly for longer
        if date_range_days <= 365 * 3:
            # Quarterly marks
            mark_dates = pd.date_range(start=min_date, end=max_date, freq='QS')
        elif date_range_days <= 365 * 10:
            # 6-monthly marks
            # Create a custom range starting from min_date and adding 6 months each time
            mark_dates = []
            current_date = min_date
            while current_date <= max_date:
                mark_dates.append(current_date)
                current_date = current_date + relativedelta(months=6)
            mark_dates = pd.DatetimeIndex(mark_dates)
        else:
            # Yearly marks
            mark_dates = pd.date_range(start=min_date, end=max_date, freq='YS')
        
        # Add marks for each date in the range
        for date in mark_dates:
            mark_value = (date - min_date).days
            
            # Format date as YYYY-MM
            date_label = date.strftime('%Y-%m')
            
            # Style the marks
            slider_marks[mark_value] = {
                'label': date_label,
                'style': {'transform': 'rotate(45deg)', 'white-space': 'nowrap'}
            }
        
        # Always include min and max dates with special styling
        slider_marks[slider_min] = {
            'label': min_date.strftime('%Y-%m'),
            'style': {'color': '#2196F3', 'font-weight': 'bold', 'transform': 'rotate(45deg)'}
        }
        slider_marks[slider_max] = {
            'label': max_date.strftime('%Y-%m'),
            'style': {'color': '#2196F3', 'font-weight': 'bold', 'transform': 'rotate(45deg)'}
        }
        
        # Initialize storage data
        storage_data = {
            "min_date": min_date_str,
            "max_date": max_date_str,
            "start_date": min_date_str,
            "end_date": max_date_str
        }
    # Update storage based on slider change
    elif trigger_id == 'date-filter-slider.value':
        try:
            storage_data = json.loads(date_filter_storage)
            
            # Ensure we have min_date
            if "min_date" not in storage_data:
                min_date_str, max_date_str = get_review_date_range()
                if min_date_str is None:
                    return json.dumps({"start_date": None, "end_date": None}), slider_min, slider_max, slider_value, slider_marks
                storage_data["min_date"] = min_date_str
                storage_data["max_date"] = max_date_str
            
            # Calculate new dates based on slider values
            min_date = datetime.strptime(storage_data["min_date"], '%Y-%m-%d')
            start_date = (min_date + timedelta(days=date_slider_value[0])).strftime('%Y-%m-%d')
            end_date = (min_date + timedelta(days=date_slider_value[1])).strftime('%Y-%m-%d')
            
            # Update only what changed
            storage_data["start_date"] = start_date
            storage_data["end_date"] = end_date
        except Exception as e:
            print(f"Error updating date filter: {str(e)}")
            return json.dumps({"start_date": None, "end_date": None}), slider_min, slider_max, slider_value, slider_marks
    else:
        # For other triggers, just parse existing storage
        storage_data = json.loads(date_filter_storage) if date_filter_storage else {"start_date": None, "end_date": None}
        
    # Return the updated storage and slider settings
    return json.dumps(storage_data), slider_min, slider_max, slider_value, slider_marks

def update_date_display(slider_value, date_filter_storage, language, total_reviews):
    """
    Generate date display HTML based on slider values and filter storage
    
    Args:
        slider_value: Current date slider value
        date_filter_storage: Current date filter storage JSON
        language: Current UI language
        total_reviews: Total number of reviews for the current filter
        translations: Dictionary of translations
        
    Returns:
        html.Div component with date range display
    """
    if not slider_value or not date_filter_storage:
        return ""
        
    try:
        # Parse the date storage
        storage = json.loads(date_filter_storage)
        min_date = storage.get("min_date")
        
        if not min_date:
            return ""
            
        # Convert slider values to dates
        min_date_obj = datetime.strptime(min_date, '%Y-%m-%d')
        start_date_obj = min_date_obj + timedelta(days=slider_value[0])
        end_date_obj = min_date_obj + timedelta(days=slider_value[1])
        
        # Format as YYYY-MM for display
        start_date_display = start_date_obj.strftime('%Y-%m')
        end_date_display = end_date_obj.strftime('%Y-%m')
        
        # Calculate the span in months for additional info
        month_diff = (end_date_obj.year - start_date_obj.year) * 12 + (end_date_obj.month - start_date_obj.month)
        
        month_label = TRANSLATIONS[language]['months']

        return html.Div([
            html.Div([
                html.Button(
                    TRANSLATIONS[language]['selected_date_range'],
                    id='apply-date-btn',
                    style={'fontSize': '14px', 'padding': '2px 8px', 'marginRight': '10px'}
                ),
                dcc.Input(
                    id='start-date-input',
                    type='text',
                    value=start_date_display,
                    style={'width': '80px', 'textAlign': 'center', 'color': '#2196F3', 'fontWeight': 'bold'}
                ),
                html.Span(" → ", style={'margin': '0 10px'}),
                dcc.Input(
                    id='end-date-input',
                    type='text',
                    value=end_date_display,
                    style={'width': '80px', 'textAlign': 'center', 'color': '#2196F3', 'fontWeight': 'bold'}
                ),
                html.Span(f" ({month_diff+1} {month_label}, {TRANSLATIONS[language]['total_reviews']}: {total_reviews})"),
                
            ], style={'marginTop': '5px'})
        ])
    except Exception as e:
        print(f"Error updating date display: {str(e)}")
        return ""

def update_date_from_input(n_clicks, start_date_input, end_date_input, date_filter_storage):
    """
    Update date filter based on manual input values
    
    Args:
        n_clicks: Number of clicks on apply button
        start_date_input: User-entered start date
        end_date_input: User-entered end date
        date_filter_storage: Current date filter storage JSON
        
    Returns:
        Tuple of (new_slider_value, updated_date_storage)
    """
    if not n_clicks or not date_filter_storage:
        raise dash.exceptions.PreventUpdate
        
    try:
        # Parse current date storage
        storage_data = json.loads(date_filter_storage)
        min_date_str = storage_data.get("min_date")
        max_date_str = storage_data.get("max_date")
        
        if not min_date_str or not max_date_str:
            raise dash.exceptions.PreventUpdate
            
        # Parse base min date
        min_date = datetime.strptime(min_date_str, '%Y-%m-%d')
        
        # Parse user input dates
        # Add day component if missing
        if len(start_date_input) <= 7:  # Format is YYYY-MM
            start_date_input = f"{start_date_input}-01"
        if len(end_date_input) <= 7:  # Format is YYYY-MM
            # Set to last day of the month
            year, month = end_date_input.split('-')
            last_day = pd.Period(f"{year}-{month}").days_in_month
            end_date_input = f"{end_date_input}-{last_day}"
            
        try:
            # Try to parse with different formats
            try:
                start_date = datetime.strptime(start_date_input, '%Y-%m-%d')
            except ValueError:
                start_date = datetime.strptime(start_date_input, '%Y-%m')
                start_date = datetime(start_date.year, start_date.month, 1)
                
            try:
                end_date = datetime.strptime(end_date_input, '%Y-%m-%d')
            except ValueError:
                end_date = datetime.strptime(end_date_input, '%Y-%m')
                # Set to last day of month
                if end_date.month == 12:
                    end_date = datetime(end_date.year + 1, 1, 1) - timedelta(days=1)
                else:
                    end_date = datetime(end_date.year, end_date.month + 1, 1) - timedelta(days=1)
        except ValueError:
            # If parsing fails, don't update
            print(f"Invalid date format: {start_date_input} or {end_date_input}")
            raise dash.exceptions.PreventUpdate
            
        # Calculate slider values
        start_slider_value = (start_date - min_date).days
        end_slider_value = (end_date - min_date).days
        
        # Ensure dates are within valid range
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
        start_slider_value = max(0, min(start_slider_value, (max_date - min_date).days))
        end_slider_value = max(start_slider_value, min(end_slider_value, (max_date - min_date).days))
        
        # Update storage
        storage_data["start_date"] = start_date.strftime('%Y-%m-%d')
        storage_data["end_date"] = end_date.strftime('%Y-%m-%d')
        
        return [start_slider_value, end_slider_value], json.dumps(storage_data)
        
    except Exception as e:
        print(f"Error updating dates from input: {str(e)}")
        raise dash.exceptions.PreventUpdate