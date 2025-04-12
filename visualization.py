import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from collections import Counter
import re
from io import BytesIO
import base64

# Set matplotlib style
plt.style.use('ggplot')
sns.set_style("whitegrid")

def create_rating_chart(df):
    """
    Create a bar chart for rating distribution
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Count ratings
    rating_counts = df['rating'].value_counts().sort_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        marker_color=['#EF4444', '#F59E0B', '#F59E0B', '#10B981', '#10B981'],
        text=rating_counts.values,
        textposition='auto'
    ))
    
    # Customize layout
    fig.update_layout(
        title='Rating Distribution',
        xaxis_title='Rating',
        yaxis_title='Number of Reviews',
        template='plotly_white',
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['1 ★', '2 ★', '3 ★', '4 ★', '5 ★']
        ),
        height=400
    )
    
    return fig

def create_sentiment_pie(df):
    """
    Create a pie chart for sentiment distribution
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Count sentiments
    sentiment_counts = df['sentiment'].value_counts()
    
    # Create color map
    color_map = {
        'positive': '#10B981',
        'neutral': '#F59E0B',
        'negative': '#EF4444'
    }
    
    # Create figure
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map=color_map,
        hole=0.4
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        legend_title=None,
        height=400
    )
    
    # Add percentage labels
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_category_bar(df):
    """
    Create a bar chart for review categories
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Count categories
    category_counts = df['category'].value_counts().head(10)  # Top 10 categories
    
    # Create figure
    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        title='Top Review Categories',
        labels={'x': 'Category', 'y': 'Number of Reviews'},
        color=category_counts.values,
        color_continuous_scale=['#DBEAFE', '#1E3A8A']
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        height=400
    )
    
    return fig

def create_wordcloud(df):
    """
    Create a word cloud from review text
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Matplotlib figure object
    """
    # Combine all review text
    text = ' '.join(df['text'].dropna())
    
    # Clean text
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Common Words in Reviews', fontsize=16)
    
    return fig

def create_language_pie(df):
    """
    Create a pie chart for language distribution
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Count languages
    language_counts = df['language'].value_counts().head(10)  # Top 10 languages
    
    # Create figure
    fig = px.pie(
        values=language_counts.values,
        names=language_counts.index,
        title='Reviews by Language',
        hole=0.4
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        legend_title=None,
        height=400
    )
    
    # Add percentage labels
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_timeline_chart(df):
    """
    Create a line chart for reviews over time
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and sentiment
    df_daily = df.groupby([pd.Grouper(key='date', freq='D'), 'sentiment']).size().reset_index(name='count')
    
    # Create figure
    fig = px.line(
        df_daily,
        x='date',
        y='count',
        color='sentiment',
        title='Reviews Timeline',
        labels={'date': 'Date', 'count': 'Number of Reviews'},
        color_discrete_map={
            'positive': '#10B981',
            'neutral': '#F59E0B',
            'negative': '#EF4444'
        }
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        xaxis_tickangle=-45,
        legend_title=None,
        height=400
    )
    
    return fig

def create_store_comparison(df):
    """
    Create a comparison chart between app stores
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Group by store and sentiment
    store_sentiment = df.groupby(['store', 'sentiment']).size().reset_index(name='count')
    
    # Create figure
    fig = px.bar(
        store_sentiment,
        x='store',
        y='count',
        color='sentiment',
        title='App Store Comparison',
        labels={'store': 'Store', 'count': 'Number of Reviews'},
        color_discrete_map={
            'positive': '#10B981',
            'neutral': '#F59E0B',
            'negative': '#EF4444'
        },
        barmode='group'
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        legend_title=None,
        height=400
    )
    
    return fig

def create_rating_by_category(df):
    """
    Create a box plot of ratings by category
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Get top categories
    top_categories = df['category'].value_counts().head(6).index
    category_df = df[df['category'].isin(top_categories)]
    
    # Create figure
    fig = px.box(
        category_df,
        x='category',
        y='rating',
        title='Rating Distribution by Category',
        labels={'category': 'Category', 'rating': 'Rating'},
        color='category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        showlegend=False,
        height=400,
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            range=[0.5, 5.5]
        )
    )
    
    return fig

def create_version_comparison(df):
    """
    Create a chart comparing ratings across app versions
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Check if app_version column exists
    if 'app_version' not in df.columns:
        return None
    
    # Get top versions
    top_versions = df['app_version'].value_counts().head(8).index
    version_df = df[df['app_version'].isin(top_versions)]
    
    # Group by version
    version_ratings = version_df.groupby('app_version')['rating'].mean().reset_index()
    version_counts = version_df.groupby('app_version').size().reset_index(name='count')
    version_data = pd.merge(version_ratings, version_counts, on='app_version')
    
    # Sort by version if possible
    try:
        version_data['version_num'] = version_data['app_version'].str.extract(r'(\d+\.\d+)')
        version_data['version_num'] = pd.to_numeric(version_data['version_num'])
        version_data = version_data.sort_values('version_num')
    except:
        # If version extraction fails, use original order
        pass
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for average rating
    fig.add_trace(go.Bar(
        x=version_data['app_version'],
        y=version_data['rating'],
        name='Avg Rating',
        marker_color='#1E3A8A',
        yaxis='y'
    ))
    
    # Add line chart for review count
    fig.add_trace(go.Scatter(
        x=version_data['app_version'],
        y=version_data['count'],
        name='Review Count',
        marker_color='#EF4444',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    # Customize layout
    fig.update_layout(
        title='Ratings by App Version',
        template='plotly_white',
        xaxis_title='App Version',
        yaxis=dict(
            title='Average Rating',
            range=[0, 5.5],
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            side='left'
        ),
        yaxis2=dict(
            title='Number of Reviews',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400
    )
    
    return fig

def create_sentiment_trend(df):
    """
    Create a stacked area chart showing sentiment trends over time
    
    Args:
        df (DataFrame): DataFrame containing review data
        
    Returns:
        Figure: Plotly figure object
    """
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and sentiment
    df_weekly = df.groupby([pd.Grouper(key='date', freq='W'), 'sentiment']).size().unstack(fill_value=0)
    
    # Create normalized percentage data
    df_weekly_pct = df_weekly.divide(df_weekly.sum(axis=1), axis=0) * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add area traces for each sentiment
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in df_weekly_pct.columns:
            fig.add_trace(go.Scatter(
                x=df_weekly_pct.index,
                y=df_weekly_pct[sentiment],
                name=sentiment.capitalize(),
                mode='lines',
                stackgroup='one',
                groupnorm='percent',
                line=dict(
                    width=0.5,
                    color={
                        'positive': '#10B981',
                        'neutral': '#F59E0B',
                        'negative': '#EF4444'
                    }.get(sentiment)
                ),
                fillcolor={
                    'positive': 'rgba(16, 185, 129, 0.6)',
                    'neutral': 'rgba(245, 158, 11, 0.6)',
                    'negative': 'rgba(239, 68, 68, 0.6)'
                }.get(sentiment)
            ))
    
    # Customize layout
    fig.update_layout(
        title='Sentiment Trend Over Time',
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Percentage',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400
    )
    
    return fig
