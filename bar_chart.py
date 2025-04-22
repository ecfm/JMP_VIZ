"""
Bar chart functionality for the visualization app.
Contains functions for creating and processing bar charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from config import TRANSLATIONS, type_colors
from data import get_bar_chart_data
from utils import ratio_to_rgb, format_category_display, get_options

def create_bar_chart(categories, counts, sentiment_ratios, colors, bar_zoom, language, original_categories):
    """Create a bar chart figure based on the provided data using a single trace."""
    fig = go.Figure()

    # Prepare data for a single trace
    # `categories` are already formatted display names (without HTML spans at this point)
    x_values = categories
    y_values = counts
    marker_colors = [ratio_to_rgb(s) for s in sentiment_ratios]
    # Customdata should be a list of lists/tuples, one for each bar
    # Add the clean category name (from `categories`) as the 4th element for hover text
    custom_data_list = [[count, sentiment, orig_cat, clean_cat]
                       for count, sentiment, orig_cat, clean_cat
                       in zip(counts, sentiment_ratios, original_categories, categories)]

    # Add the single Bar trace
    fig.add_trace(go.Bar(
        x=x_values,
        y=y_values,
        marker=dict(
            color=marker_colors, # List of colors for each bar
            line=dict(
                color='rgba(0,0,0,0.2)',
                width=1
            )
        ),
        customdata=custom_data_list, # List of custom data for each bar
        hovertemplate=(
            f"<b>%{{customdata[3]}}</b><br>" + # Use index 3 for the clean category name
            f"{TRANSLATIONS[language]['hover_count']}: %{{customdata[0]}}<br>" +
            f"{TRANSLATIONS[language]['hover_satisfaction']}: %{{customdata[1]:.2f}}<extra></extra>" # Use customdata indices
        ),
        showlegend=False
    ))

    # Add an invisible satisfaction heatmap for color legend
    # This part might still cause issues, but let's keep it for now.
    # Consider removing it if hover problems persist after switching to single trace.
    if sentiment_ratios: # Only add if there's data
        all_sentiments = np.array(sentiment_ratios).reshape(-1, 1)
        fig.add_trace(go.Heatmap(
            z=all_sentiments,
            colorscale=px.colors.diverging.RdBu,
            showscale=True,
            opacity=0, # Keep it invisible
            zmin=0, # Explicitly set range if needed
            zmax=1,
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
            ),
            hoverinfo='skip' # Disable hover for the heatmap itself (changed from 'none')
        ))

    # Create custom x-axis ticktext with colored category labels
    ticktext = []
    is_zoomed = bar_zoom and bar_zoom != 'all'

    for i, category in enumerate(categories):
        category_color = None
        try:
            original_cat_for_tick = original_categories[i]
            for prefix, color_value in type_colors.items():
                if original_cat_for_tick.startswith(prefix):
                    category_color = color_value
                    break
        except IndexError:
            pass

        if category_color:
            ticktext.append(f'<span style="color:{category_color}; font-weight:bold">{category}</span>')
        else:
            ticktext.append(category)

    # Update layout for bar chart
    title_key = 'bar_chart_title'
    fig.update_layout(
        title=TRANSLATIONS[language][title_key],
        xaxis=dict(
            title=None,
            tickangle=20,
            tickmode='array',
            tickvals=list(range(len(categories))), # Use integer indices for tickvals
            ticktext=ticktext, # Use the generated HTML strings for ticktext
            # Explicitly set categoryorder to match the input data order
            categoryorder='array',
            categoryarray=categories, # Provide the category names in order
            domain=[0, 0.85]
        ),
        yaxis=dict(
            title=TRANSLATIONS[language]['hover_count'],
        ),
        showlegend=False,
        hovermode='closest', # Ensure hover focuses on the nearest bar
        height=700,
        width=1200,
        margin=dict(
            l=80,
            r=150,
            t=100,
            b=180,
            autoexpand=True
        ),
    )

    # Adjust the chart height based on number of categories
    if len(categories) > 15:
        fig.update_layout(height=750)
    elif len(categories) > 10:
        fig.update_layout(height=720)
    
    # Update bar width based on number of categories
    if len(categories) <= 5:
        fig.update_layout(bargap=0.5)
    elif len(categories) <= 10:
        fig.update_layout(bargap=0.3)
    else:
        fig.update_layout(bargap=0.1)
    
    # Create a custom legend as an annotation
    legend_items = []
    present_prefixes = set()
    for orig_cat in original_categories:
        for prefix in type_colors:
            if orig_cat.startswith(prefix):
                present_prefixes.add(prefix)
                break

    for prefix, color in type_colors.items():
        if prefix in present_prefixes:
            translated_label = TRANSLATIONS[language]['usage_category'] if prefix == 'U: ' else (
                TRANSLATIONS[language]['attribute_category'] if prefix == 'A: ' else TRANSLATIONS[language]['performance_category']
            )
            legend_items.append(f'<span style="color:{color}; font-weight:bold; margin-right:15px;">▣ {translated_label}</span>')
    
    if legend_items:
        legend_title = TRANSLATIONS[language].get('category_types', 'Category Types')
        legend_text = f'<b>{legend_title}:</b> ' + ' '.join(legend_items)
        
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=1.2,
            y=-0.22,
            text=legend_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.5)",
            borderwidth=1,
            borderpad=6,
            align="center"
        )
    
    return fig

def process_bar_chart_data(bar_categories, bar_zoom_value, bar_count, language, search_query, start_date, end_date):
    """
    Process data for bar chart visualization and controls.
    
    Args:
        bar_categories: List of category types to include
        bar_zoom_value: Value of the selected zoom category
        bar_count: Maximum number of bars to display
        language: Current language setting
        search_query: Search query for filtering
        start_date: Start date for filtering
        end_date: End date for filtering
        
    Returns:
        Tuple containing (figure, zoom_dropdown_options, new_zoom_value, updated_bar_count)
    """
    # Handle empty values
    if not bar_categories:
        bar_categories = ['usage', 'attribute', 'performance']
    if bar_zoom_value is None or bar_zoom_value == '': # Ensure empty string is treated same as None
        bar_zoom_value = 'all'
    # Get bar chart data
    display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
        bar_categories, bar_zoom_value, language, search_query, start_date, end_date
    )
    
    # Update bar count based on available categories
    bar_count = min(bar_count or 10, len(display_categories))
    
    # Apply bar count limit consistently to all lists needed for display
    if bar_count > 0 and bar_count < len(display_categories):
        display_categories_viz = display_categories[:bar_count]
        original_categories_viz = original_categories[:bar_count] # Truncate original categories
        display_counts_viz = counts[:bar_count]
        display_sentiment_ratios_viz = sentiment_ratios[:bar_count]
        display_colors_viz = colors[:bar_count]
    else:
        display_categories_viz = display_categories
        original_categories_viz = original_categories # Use full list
        display_counts_viz = counts
        display_sentiment_ratios_viz = sentiment_ratios
        display_colors_viz = colors

    # Format categories for display labels and dropdowns
    formatted_categories_viz = [format_category_display(category, language) for category in display_categories_viz]
    # Dropdown options should reflect the displayed categories
    dropdown_options = get_options(language, bar_zoom_value, original_categories_viz, formatted_categories_viz)

    # Create the bar chart figure, passing the truncated/full lists including original_categories_viz
    fig = create_bar_chart(
        formatted_categories_viz, # Pass formatted categories for display
        display_counts_viz,
        display_sentiment_ratios_viz,
        display_colors_viz,
        bar_zoom_value,
        language,
        original_categories_viz # Pass corresponding original categories
    )
    return fig, dropdown_options, bar_zoom_value, bar_count 