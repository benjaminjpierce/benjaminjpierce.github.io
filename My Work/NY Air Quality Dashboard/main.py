import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

def main():

    # load data
    df = pd.read_csv('Air_quality.csv')

    # extract year from "Start_Date" column for slider
    df['Year'] = pd.to_datetime(df['Start_Date']).dt.year

    # define dash app object
    app = dash.Dash(__name__)

    # define layout of dashboard
    app.layout = html.Div([

        # header
        html.H1("New York City Air Quality Dashboard"),

        # markdown component for additional information
        dcc.Markdown("""
        ## Explore New York City air quality data using interactive dashboard components
        ### *Select an indicator, borough, and year to get started*
        """),

        # space between next component
        html.Br(),

        # radio button component to select indicator, filter to only allow primary air quality indicators as options
        dcc.RadioItems(
            id='indicator-radio',
            options=[{'label': indicator, 'value': indicator} for indicator in
                     ['Ozone (O3)', 'Sulfur Dioxide (SO2)', 'Nitrogen Dioxide (NO2)']],
            value='Ozone (O3)'
        ),

        html.Br(),
        html.Br(),

        # dropdown component to select geographic area (boroughs only)
        dcc.Dropdown(
            id='geo-dropdown',
            options=[{'label': geo, 'value': geo} for geo in
                     df[df['Geo Type Name'] == 'Borough']['Geo Place Name'].unique()],
            value=df[df['Geo Type Name'] == 'Borough']['Geo Place Name'].iloc[0]
        ),

        html.Br(),
        html.Br(),

        # slider component to select year
        dcc.Slider(
            id='year-slider',
            min=df['Year'].min(),
            max=df['Year'].max(),
            step=1,
            marks={str(year): str(year) for year in df['Year'].unique()},
            value=df['Year'].iloc[0],
            included=False
        ),

        html.Br(),

        # display measurement information for selected indicator
        html.Div([
            html.H3("Selected Indicator Info:"),
            html.Div(id='selected-indicator-info'),
        ]),

        # bar chart component to show selected indicator by location for given year
        dcc.Graph(id='bar-chart'),

    ])


    # define callback function to update displayed indicator information based on user input
    @app.callback(
        Output('bar-chart', 'figure'),
        Output('selected-indicator-info', 'children'),
        Input('indicator-radio', 'value'),
        Input('geo-dropdown', 'value'),
        Input('year-slider', 'value')
    )


    def update_visualizations(selected_indicator, selected_geo, selected_year):
        """
        update visualizations based on user input

        args:
        - selected_indicator (str): selected air quality indicator
        - selected_geo (str): selected geographic area
        - selected_year (int): selected year

        returns:
        - (dict): Plotly bar chart figure
        - (str): indicator measurement info
        """

        filtered_df = df[
            (df['Name'] == selected_indicator) & (df['Geo Place Name'] == selected_geo) & (df['Year'] == selected_year)]

        if len(filtered_df) == 0:
            return {}, "Indicator info not available for the selected options."

        # calculate max value for selected indicator across all years with the same indicator
        max_range = df[(df['Name'] == selected_indicator)]['Data Value'].max() * 1.5

        # create a fixed y-axis for consistency
        y_range = [0, max_range]

        # create a bar chart
        bar_chart = px.bar(filtered_df, x='Time Period', y='Data Value', title=f'{selected_indicator} in {selected_geo}')
        bar_chart.update_yaxes(range=y_range)

        # get measurement info for the selected indicator (if available)
        indicator_info = filtered_df.iloc[0]
        measure = indicator_info['Measure']
        measure_info = indicator_info['Measure Info']

        return bar_chart, f'Measure: {measure} in {measure_info}'


    if __name__ == '__main__':
        app.run_server(debug=True)


if __name__ == '__main__':
    main()
