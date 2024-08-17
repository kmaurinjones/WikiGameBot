import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

def line_plot(game_csv):
    """Create a line plot of similarity over time. Uses a polynomial trend line."""
    # Original line plot
    fig = px.line(game_csv, x='turn', y='similarity_to_target', markers=True,
                  title='Line Plot of Similarity Over Time',
                  labels={'turn': 'Game Turn Number', 'similarity_to_target': 'Similarity to Target Topic'})

    fig.update_traces(hovertemplate='Wiki Page Number: %{x}<br>Similarity: %{y}<br>Page Title: %{customdata[0]}<br>Turn Time (seconds): %{customdata[1]}',
                      customdata=game_csv[['current_topic', 'turn_time']])

    # Polynomial trend line
    x = game_csv['turn']
    y = game_csv['similarity_to_target']
    z = np.polyfit(x, y, 2)  # Second-degree polynomial fit (quadratic)
    p = np.poly1d(z)

    # Generate trend line values
    trend_x = np.linspace(x.min(), x.max(), 100)
    trend_y = p(trend_x)

    # Add the trend line to the plot
    fig.add_scatter(x=trend_x, y=trend_y,
                    mode='lines', 
                    line=dict(dash='dot', color='rgba(255, 105, 180, 0.8)', width=3))  # Complementary color with less opacity, thicker line

    # Update the layout for the title, background colors, and hide the legend
    fig.update_layout({
        'title': {
            'text': 'Similarity Over Game Turns',
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        },
        'showlegend': False  # Hide the legend
    })

    st.plotly_chart(fig)  # Use Streamlit's function to display Plotly chart

def plot_topic_clusters(game_csv):
    """Create a 2D scatter plot of topics based on their embeddings, with similarity as point size."""
    # Extract embeddings, topics, and similarities from the game_csv
    embeddings = game_csv['embedding'].tolist()
    topics = game_csv['current_topic'].tolist()
    similarities = game_csv['similarity_to_target'].tolist()

    # Convert embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Check if we have enough data points for clustering
    n_samples = len(embeddings_array)
    if n_samples < 2:
        st.warning("Not enough topics to create a meaningful cluster plot.")
        return

    # Adjust perplexity based on the number of samples
    perplexity = min(30, n_samples - 1)

    # Perform dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    topic_positions = tsne.fit_transform(embeddings_array)

    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': topic_positions[:, 0],
        'y': topic_positions[:, 1],
        'topic': topics,
        'similarity': similarities
    })

    # Normalize similarities for point size (larger values for higher similarity)
    df_plot['point_size'] = (df_plot['similarity'] - df_plot['similarity'].min()) / (df_plot['similarity'].max() - df_plot['similarity'].min())
    df_plot['point_size'] = df_plot['point_size'] * 30 + 10  # Scale to a reasonable size range

    # Create an interactive scatter plot using Plotly with color gradient for similarity
    fig = px.scatter(
        df_plot, 
        x='x', 
        y='y', 
        hover_data=['topic', 'similarity'],
        text='topic',
        size='point_size',
        color='similarity',  # Color points by similarity
        color_continuous_scale=px.colors.sequential.Viridis,  # Color gradient from 0 to 1
        title='Topic Relationship and Similarity Map',
        labels={
            'x': '',  # Hide x-axis label
            'y': '',  # Hide y-axis label
            'similarity': 'Similarity to Target'
        }
    )

    # Update layout to hide axis labels and ticks, show color bar, and center the title
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        coloraxis_colorbar=dict(title="Similarity to Target"),  # Add color bar with title
        title={'text': 'Topic Relationship and Similarity Map', 'x': 0.5, 'xanchor': 'center'},
        font=dict(size=12)
    )

    # Update traces to show topic names
    fig.update_traces(textposition='top center')

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
