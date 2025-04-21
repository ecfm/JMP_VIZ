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

def create_bar_chart(categories, counts, sentiment_ratios, colors, bar_zoom, language):
    """Create a bar chart figure based on the provided data."""
    # Create the bar chart figure
    fig = go.Figure()
    
    # Add bars
    for i, (category, count, sentiment, color) in enumerate(zip(categories, counts, sentiment_ratios, colors)):
        # Use the sentiment color for the bar fill
        fig.add_trace(go.Bar(
            x=[category],
            y=[count],
            name=category,
            marker=dict(
                color=ratio_to_rgb(sentiment),  # Use sentiment color for bar
                line=dict(
                    color='rgba(0,0,0,0.2)',  # Light gray border
                    width=1
                )
            ),
            hovertemplate=(
                f"{TRANSLATIONS[language]['hover_count']}: %{{y}}<br>" +
                f"{TRANSLATIONS[language]['hover_satisfaction']}: {sentiment:.2f}"
            ),
            showlegend=False  # Hide individual bars from legend
        ))
    
    # Add an invisible satisfaction heatmap for color legend
    all_sentiments = np.array(sentiment_ratios).reshape(-1, 1)
    fig.add_trace(go.Heatmap(
        z=all_sentiments,
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
    
    # Create custom x-axis ticktext with colored category labels
    ticktext = []
    
    # Check if we're zoomed into a parent category
    is_zoomed = bar_zoom and bar_zoom != ''
    
    for category in categories:
        # Determine the color based on the category prefix
        category_color = None
        for prefix, color_value in type_colors.items():
            if category.startswith(prefix):
                category_color = color_value
                break
        
        # Format display text with colored labels in HTML
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
            tickvals=list(range(len(categories))),
            ticktext=ticktext,
            domain=[0, 0.85]
        ),
        yaxis=dict(
            title=TRANSLATIONS[language]['hover_count'],
        ),
        showlegend=False,
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
    for prefix, color in type_colors.items():
        if any(cat.startswith(prefix) for cat in categories):
            translated_label = TRANSLATIONS[language]['usage_category'] if prefix == 'U: ' else (
                TRANSLATIONS[language]['attribute_category'] if prefix == 'A: ' else TRANSLATIONS[language]['performance_category']
            )
            if not translated_label.startswith(prefix):
                translated_label = prefix + translated_label
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
    if bar_zoom_value is None:
        bar_zoom_value = 'all'
    # Get bar chart data
    display_categories, original_categories, counts, sentiment_ratios, review_data, colors, title_key = get_bar_chart_data(
        bar_categories, bar_zoom_value, language, search_query, start_date, end_date
    )
    
    # Update bar count based on available categories
    bar_count = min(bar_count or 10, len(display_categories))
    
    # Apply bar count limit from bar_count slider for visualization
    if bar_count > 0 and bar_count < len(display_categories):
        display_categories = display_categories[:bar_count]
        original_categories = original_categories[:bar_count]
        display_counts = counts[:bar_count]
        display_sentiment_ratios = sentiment_ratios[:bar_count]
        display_colors = colors[:bar_count]
    else:
        display_counts = counts
        display_sentiment_ratios = sentiment_ratios
        display_colors = colors

    formatted_categories = [format_category_display(category, language) for category in display_categories]
    dropdown_options = get_options(language, bar_zoom_value, original_categories, formatted_categories)
    # Create the bar chart figure
    fig = create_bar_chart(formatted_categories, display_counts, display_sentiment_ratios, display_colors, bar_zoom_value, language)
    return fig, dropdown_options, bar_zoom_value, bar_count 