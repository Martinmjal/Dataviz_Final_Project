# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Read the data once at the start of the app to avoid re-reading it in each callback
df = pd.read_csv('churn_prediction_features.csv')

# Determining which are the numerical columns
cols_to_drop = [col for col in df.columns if col.startswith('sub_categoria_') or col.startswith('warehouse_')]
cols_to_drop.extend(['customer_id', 'has_churned_before', 'is_churned'])
numerical_cols = df.drop(columns=cols_to_drop, axis=1).select_dtypes(include=['float64', 'int64']).columns

# Initialize the Dash app with suppress_callback_exceptions set to True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Custom CSS styles
styles = {
    'main_container': {
        'fontFamily': 'Arial, sans-serif',
        'margin': 'auto',
        'width': '80%',
        'padding': '20px'
    },
    'header': {
        'textAlign': 'center',
        'color': '#007bff'
    },
    'tab': {
        'borderRadius': '5px',
        'padding': '10px',
        'margin': '10px',
        'backgroundColor': '#f8f9fa',
        'fontFamily': 'Arial, sans-serif',
        'fontSize': '20px'
    },
    'graph_container': {
        'display': 'inline-block',
        'width': '50%',
        'padding': '10px'
    },
    'dropdown': {
        'marginBottom': '10px'
    },
    'checkbox': {
        'margin': '10px'
    }
}

# Define the layout of the app
app.layout = html.Div([
    html.H1('Customer Classification Model: Churn Prediction', 
            style=styles['header']),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='EDA', value='tab-1', style=styles['tab']),
        dcc.Tab(label='Classifiers', value='tab-2', style=styles['tab']),
    ]),
    
    html.Div(id='tabs-content')
], style=styles['main_container'])

# Callback for rendering tabs content
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        # Chart 1: Class count in a bar chart
        category_counts = df['is_churned'].replace({0: 'not churned', 1: 'churned'}).value_counts()
        fig1 = px.bar(category_counts, x=category_counts.index, y=category_counts.values)
        fig1.update_layout(
            yaxis_title='',
            xaxis_title='',
            title={'text': 'Count of churned and non churned customers', 'x': 0.5, 'xanchor': 'center'}
        )

        # Chart 2: Table with the null percentages and a color scale
        null_percentage = df.isnull().sum() / len(df) * 100
        null_df = pd.DataFrame(null_percentage, columns=['missing_percentage'])
        colors = ['rgb(255, 0, 0)' if p >= 50 else f'rgb({255}, {255 - int(p/50*255)}, {255 - int(p/50*255)})' for p in null_df['missing_percentage']]
        
        fig2 = go.Figure(data=[go.Table(
            header=dict(values=['Column', 'Missing Percentage'], align='center', font=dict(color='black', size=12)),
            cells=dict(values=[null_df.index, null_df['missing_percentage']], fill_color=[['white'], colors], align='center', font=dict(color='black', size=11))
        )])
        fig2.update_layout(title='Null Values Percentage per Column with Gradient Color', title_x=0.5)

        # Layout for the boxplot chart
        boxplot_layout = html.Div([
            html.H3("Feature Distribution Analysis"),
            dcc.Dropdown(
                id='numerical-dropdown',
                options=[{'label': col, 'value': col} for col in numerical_cols],
                value=numerical_cols[0],  # set the default value to the first numerical column
                style=styles['dropdown']
            ),
            dcc.Checklist(
                id='outlier-checkbox',
                options=[{'label': 'Include Outliers', 'value': 'include_outliers'}],
                value=[],
                style=styles['checkbox']
            ),
            html.P("Outliers are defined as data points that are below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, "
                   "where Q1 and Q3 are the first and third quartiles, respectively, and IQR is the interquartile range.",
                   style={'margin-top': '10px'}),
            dcc.Graph(id='boxplot-graph'),
        ])

        return html.Div([
            html.Div([dcc.Graph(figure=fig1)], style=styles['graph_container']),
            html.Div([dcc.Graph(figure=fig2)], style=styles['graph_container']),
            boxplot_layout
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.H5('The model metrics and predictions will go here')
        ])

# Function to remove outliers
def remove_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    return dataframe[(dataframe[column] >= (Q1 - 1.5 * IQR)) & (dataframe[column] <= (Q3 + 1.5 * IQR))]

# Callback for the boxplot chart
@app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('numerical-dropdown', 'value'), Input('outlier-checkbox', 'value')]
)
def update_graph(selected_column, checkbox_values):
    include_outliers = 'include_outliers' in checkbox_values

    if include_outliers:
        # If checkbox is checked, use the original dataframe
        filtered_df = df
    else:
        # If checkbox is not checked, remove outliers
        filtered_df = remove_outliers(df, selected_column)
        
    fig = px.box(filtered_df, x=selected_column, y='is_churned')
    fig.update_traces(orientation='h')
    fig.update_layout(title=f'Boxplot of {selected_column} by Churn Status', yaxis_title=selected_column)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)