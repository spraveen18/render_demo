import pandas as pd
from datetime import datetime
import re
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as px
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

    html.H2("Tweet Count by Day of Week"),
    dcc.Graph(id='tweet-count-by-day')
])

# Callbacks to update the graphs based on the filter selections
@app.callback(
    Output('trend-from-date', 'figure'),
    Output('sentiment-by-country', 'figure'),
    Output('heatmap', 'figure'),
    Output('tweet-count-by-day', 'figure'),
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

    # Create updated figures based on the filtered data
    fig1 = px.line(daily_tweet_count, x='date', y='tweet_count', title='Trend of Tweet Count over Time')
    fig2 = px.choropleth(country_sentiment, locations='country', locationmode='country names', color='avg_sentiment',
                         hover_name='country', color_continuous_scale='RdYlGn', title='Average Sentiment by Country')
    fig3 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.index.values, y=corr.columns.values))
    fig4 = px.bar(day_of_week_tweet_count, x='day_of_week', y='tweet_count', title='Tweet Count by Day of Week')

    return fig1, fig2, fig3, fig4

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
