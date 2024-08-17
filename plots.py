import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from transformers import pipeline

def line_plot(game_csv):
    """Create a line plot of similarity over time. Uses a polynomial trend line."""
    # Original line plot
    fig = px.line(game_csv, x='turn', y='similarity_to_target', markers=True,
                  title='Line Plot of Similarity Over Time',
                  labels={'turn': 'Wiki Page Number', 'similarity_to_target': 'Similarity to Target'})

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
                    line=dict(dash='dot', color='rgba(255, 105, 180, 0.8)', width=3),  # Complementary color with less opacity, thicker line
                    name='Polynomial Trend Line')

    # Update the layout for the title and the background colors
    fig.update_layout({
        'title': {
            'text': 'Similarity Over Time',
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        }
    })

    st.plotly_chart(fig)  # Use Streamlit's function to display Plotly chart

def plot_topic_clusters(game_csv):
    """Plot the topic clusters over time."""
    embeddings = np.array(game_csv['embedding'].tolist())

    # Function to calculate the optimal number of clusters
    def calculate_optimal_k(sse, K):
        second_derivative = np.diff(sse, 2)
        optimal_k = K[np.argmax(second_derivative) + 1]
        return optimal_k

    # Elbow Method to find the optimal number of clusters
    sse = []
    max_clusters = min(len(embeddings), 10)
    K = range(1, max_clusters + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

    optimal_k = calculate_optimal_k(sse, np.array(K))
    optimal_k = min(optimal_k, len(embeddings))

    # K-Means Clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    game_csv['topic'] = kmeans.fit_predict(embeddings)

    # Create a pipeline for text classification
    pipe = pipeline("text-classification", model="jonaskoenig/topic_classification_04", from_tf=True)

    # Get representative texts for each topic
    representative_texts = []
    for topic in range(optimal_k):
        sample = game_csv[game_csv['topic'] == topic].sample(1)['current_summary']
        representative_texts.append(sample.iloc[0])

    # Classify these texts to get labels
    topic_labels = [pipe(text)[0]['label'] for text in representative_texts]

    # Create a mapping from topic number to label
    topic_label_mapping = {i: label for i, label in enumerate(topic_labels)}

    # Apply this mapping to your game_csv DataFrame
    game_csv['topic_label'] = game_csv['topic'].map(topic_label_mapping)

    # Create the plot
    plot_game_csv = game_csv[['turn', 'topic_label', 'current_topic']]
    fig = px.line(plot_game_csv, x='turn', y='topic_label', markers=True, 
                  title='Topic Distribution Over Turns',
                  labels={'turn': 'Turn', 'topic_label': 'Topic'})

    fig.update_traces(hovertemplate='Turn: %{x}<br>Topic: %{y}<br>Page Title: %{customdata}<extra></extra>',
                      customdata=plot_game_csv['current_topic'])

    fig.update_layout({
        'title': {
            'text': 'Topic Distribution Over Turns',
            'x': 0.5,
            'xanchor': 'center'
        }
    })

    st.plotly_chart(fig)

# def plot_topic_clusters(game_csv):
#     # Assuming 'game_csv' contains 'embedding', 'turn', and 'text_column'

#     embeddings = pd.DataFrame(game_csv['embedding'].tolist())

#     # Function to calculate the optimal number of clusters
#     def calculate_optimal_k(sse, K):
#         # Calculate the second derivative
#         second_derivative = np.diff(sse, 2)
#         optimal_k = K[np.argmax(second_derivative) + 1]  # +1 due to the difference in indices
#         return optimal_k

#     # Elbow Method to find the optimal number of clusters
#     sse = []
#     max_clusters = min(len(embeddings), 10)  # Ensure max clusters do not exceed number of samples
#     K = range(1, max_clusters + 1)
#     for k in K:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(embeddings)
#         sse.append(kmeans.inertia_)

#     optimal_k = calculate_optimal_k(sse, np.array(K))

#     # Ensure optimal_k is not greater than the number of samples
#     optimal_k = min(optimal_k, len(embeddings))

#     # K-Means Clustering with the optimal number of clusters
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42)
#     game_csv['topic'] = kmeans.fit_predict(embeddings)

#     # Create a pipeline for text classification
#     pipe = pipeline("text-classification", model="jonaskoenig/topic_classification_04", from_tf = True)

#     # Get representative texts for each topic
#     representative_texts = []
#     for topic in range(optimal_k):
#         sample = game_csv[game_csv['topic'] == topic].sample(1)['text_column']  # Replace 'text_column' with the name of your text column
#         representative_texts.append(sample.iloc[0])

#     # Classify these texts to get labels
#     topic_labels = [pipe(text)[0]['label'] for text in representative_texts]

#     # Create a mapping from topic number to label
#     topic_label_mapping = {i: label for i, label in enumerate(topic_labels)}

#     # Apply this mapping to your game_csv DataFrame
#     game_csv['topic_label'] = game_csv['topic'].map(topic_label_mapping)

#     # Now modify the plotting code to use 'topic_label'
#     plot_game_csv = game_csv.groupby(['turn', 'topic_label']).size().reset_index(name='count')
#     fig = px.line(plot_game_csv, x='turn', y='topic_label', markers=True, 
#                   title='Topic Selection Over Time',
#                   labels={'turn': 'Turn', 'topic_label': 'Topic'})

#     fig.update_traces(hovertemplate='Turn: %{x}<br>Topic: %{y}<br>Count: %{customdata}<extra></extra>',
#                       customdata=plot_game_csv['count'])

#     fig.update_layout({
#         'title': {
#             'text': 'Topic Distribution Over Turns',
#             'x': 0.5,
#             'xanchor': 'center'
#         }
#     })

#     st.plotly_chart(fig)
