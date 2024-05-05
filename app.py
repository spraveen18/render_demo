import pandas as pd
from datetime import datetime
import re
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx 
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Load the CSV data into a DataFrame
df = pd.read_csv('drought_tweet_sentiment.csv')

df['date'] = pd.to_datetime(df['date']).dt.tz_convert(None)

# Add columns for month, year, day of the week, day name, and weekday/weekend indicators
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
df['day_name'] = df['date'].dt.day_name()
df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).map({True: 'Weekend', False: 'Weekday'})
df['hour'] = df['date'].dt.hour

# Add tweet length column
df['tweet_length'] = df['tweet'].apply(len)

# Add hashtag count column
df['hashtag_count'] = df['tweet'].apply(lambda x: len(re.findall(r'#(\w+)', x)))

# Add mention count column
df['mention_count'] = df['tweet'].apply(lambda x: len(re.findall(r'@(\w+)', x)))

# Add word count column
df['word_count'] = df['tweet'].apply(lambda x: len(x.split()))

# Perform sentiment analysis and add columns for polarity and subjectivity
df['polarity'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['subjectivity'] = df['tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the options for the filters
months = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_options = [{'label': 'Weekday', 'value': 'Weekday'}, {'label': 'Weekend', 'value': 'Weekend'}]

# Define the layout of the app
app.layout = html.Div([
    html.H1("Data Visualizations"),

    html.Div([
        html.Label("Select Month:"),
        dcc.Dropdown(
            id='month-filter',
            options=[{'label': month_name, 'value': month_num} for month_num, month_name in months.items()],
            value=None
        ),
    ]),

    html.Div([
        html.Label("Select Day of the Week:"),
        dcc.Dropdown(
            id='day-of-week-filter',
            options=[{'label': day, 'value': day} for day in days_of_week],
            value=None
        ),
    ]),

    html.Div([
        html.Label("Select Weekday/Weekend:"),
        dcc.Dropdown(
            id='weekday-weekend-filter',
            options=weekday_options,
            value=None
        ),
    ]),

    html.H2("Trend from Date"),
    dcc.Graph(id='trend-from-date'),

    html.H2("Sentiment by Country"),
    dcc.Graph(id='sentiment-by-country'),

    html.H2("Heatmap"),
    dcc.Graph(id='heatmap'),

    html.H2("Tweet Count"),
    dcc.Graph(id='tweet-count'),

    html.H2("Association Graph"),
    dcc.Graph(id='association-graph'),
])

# Callbacks to update the graphs based on the filter selections
@app.callback(
    Output('trend-from-date', 'figure'),
    Output('sentiment-by-country', 'figure'),
    Output('heatmap', 'figure'),
    Output('tweet-count', 'figure'),
    Output('association-graph', 'figure'),
    Input('month-filter', 'value'),
    Input('day-of-week-filter', 'value'),
    Input('weekday-weekend-filter', 'value')
)
def update_graphs(selected_month, selected_day_of_week, selected_weekday_weekend):
    filtered_df = df.copy()

    if selected_month:
        filtered_df = filtered_df[filtered_df['month'] == selected_month]
    if selected_day_of_week:
        filtered_df = filtered_df[filtered_df['day_of_week'] == selected_day_of_week]
    if selected_weekday_weekend:
        if selected_weekday_weekend == 'Weekday':
            filtered_df = filtered_df[filtered_df['is_weekend'] == 'Weekday']
        else:
            filtered_df = filtered_df[filtered_df['is_weekend'] == 'Weekend']

    # Recalculate necessary data for the graphs based on the filtered DataFrame
    daily_tweet_count = filtered_df.groupby('date').size().reset_index(name='tweet_count')
    country_sentiment = filtered_df.groupby('country')['polarity'].mean().reset_index(name='avg_sentiment')
    numerical_columns = ['tweet_length', 'hashtag_count', 'mention_count', 'word_count', 'polarity', 'subjectivity', 'hour']
    corr = filtered_df[numerical_columns].corr()
    day_of_week_tweet_count = filtered_df['day_of_week'].value_counts().reset_index()
    day_of_week_tweet_count.columns = ['day_of_week', 'tweet_count']

    G = nx.from_pandas_edgelist(filtered_df, 'source', 'target', create_using=nx.DiGraph)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []

    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    association_graph = go.Figure(data=[edge_trace, node_trace],
                                  layout=go.Layout(
                                      title='Association Graph',
                                      titlefont_size=16,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20, l=5, r=5, t=40),
                                      annotations=[dict(
                                          text="Python code: <a href='https://plotly.com/python/network-graphs/'> https://plotly.com/python/network-graphs/</a>",
                                          showarrow=False,
                                          xref="paper", yref="paper",
                                          x=0.005, y=-0.002)],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                                  )

    return fig1, fig2, fig3, fig4, association_graph

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
