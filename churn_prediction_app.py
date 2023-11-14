# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Read the data once at the start of the app to avoid re-reading it in each callback
df = pd.read_csv('final_dataframe_churn_prediction.csv')

df['is_churned'] = df['is_churned'].replace({0: 'not churned', 1: 'churned'})

# Determining which are the numerical columns
categorical_cols = ['warehouse_Bogota', 'sub_categoria_Ambientadores', 'sub_categoria_Complementos y vitaminas', 'is_churned']
numerical_cols = [col for col in df.columns if col not in categorical_cols]

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
        'color': '#007bff',
        'marginBottom': '20px',
        'fontFamily': 'Arial, sans-serif'
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
        'width': '48%',
        'padding': '10px'
    },
    'dropdown': {
        'marginBottom': '10px'
    },
    'checkbox': {
        'margin': '10px'
    }
}

# Define the layout of the app with additional styling
app.layout = html.Div([
    html.H1('Customer Classification Model: Churn Prediction', 
            style=styles['header']),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='EDA', value='tab-1', style=styles['tab']),  # Apply tab_style to both tabs
        dcc.Tab(label='Classifiers', value='tab-2', style=styles['tab']),
    ]),
    
    html.Div(id='tabs-content')  # Add padding to the content area
], style=styles['main_container'])

# Callback for rendering tabs content
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        # Chart 1: Class count in a bar chart
        category_counts = df['is_churned'].value_counts()
        fig1 = px.bar(category_counts, x=category_counts.index, y=category_counts.values)
        fig1.update_layout(
            yaxis_title='',
            xaxis_title='',
            title={'text': 'Count of churned and non churned customers', 'x': 0.5, 'xanchor': 'center'}
        )

        # Layout for the boxplot chart
        boxplot_layout = html.Div([
            html.H3("Feature Distribution Analysis"),
            dcc.Dropdown(
                id='numerical-dropdown',
                options= [{'label': col, 'value': col} for col in sorted(numerical_cols)],
                value=numerical_cols[0],  # set the default value to the first numerical column
                style=styles['dropdown']
            ),
            dcc.Checklist(
                id='outlier-checkbox',
                options= [{'label': 'Include Outliers', 'value': 'include_outliers'}],
                value=[],
                style=styles['checkbox']
            ),
            html.Div([
                html.Div([dcc.Graph(id='boxplot-graph')], style=styles['graph_container']),
                html.Div([dcc.Graph(id='histogram-graph')], style=styles['graph_container'])  # New graph for histogram
            ], style={'display': 'flex'})  # This ensures the boxplot and histogram are side by side
        ])
        #     html.P("Outliers are defined as data points that are below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR, "
        #            "where Q1 and Q3 are the first and third quartiles, respectively, and IQR is the interquartile range.",
        #            style={'margin-top': '10px'}),
        #     dcc.Graph(id='boxplot-graph'),
        # ])
        
        # Bar chart for comparing other variables against the is_churned class
        bar_chart_layout = html.Div([
            html.H3("Comparison of Other Variables with 'is_churned' Class"),
            dcc.Dropdown(
                id='comparison-dropdown',
                options=[{'label': col, 'value': col} for col in sorted(categorical_cols) if col not in ['is_churned', 'customer_id']],
                value= [col for col in categorical_cols if col not in ['is_churned', 'customer_id']][0]  # set the default value to the first column (excluding 'is_churned')
            ),
            dcc.Graph(id='bar-chart'),
        ])

        return html.Div([
            html.Div([dcc.Graph(figure=fig1)], style=styles['graph_container']),
            boxplot_layout,
            bar_chart_layout
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

def update_boxplot(selected_column, checkbox_values):
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

@app.callback(
    Output('histogram-graph', 'figure'),
    [Input('numerical-dropdown', 'value'), Input('outlier-checkbox', 'value')]
)
def update_histogram(selected_column, checkbox_values):
    include_outliers = 'include_outliers' in checkbox_values

    if include_outliers:
        # If checkbox is checked, use the original dataframe
        filtered_df = df
    else:
        # If checkbox is not checked, remove outliers
        filtered_df = remove_outliers(df, selected_column)

    fig = px.histogram(filtered_df, x=selected_column)
    fig.update_layout(title=f'Histogram of {selected_column}', xaxis_title=selected_column, yaxis_title='Count')
    return fig

# Callback for the bar chart comparing other variables with 'is_churned' class
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('comparison-dropdown', 'value')]
)
def update_bar_chart(selected_column):
    
    # Pivot table to get the counts
    pivot_table = df.groupby(['is_churned', selected_column]).size().unstack().fillna(0)
    # Calculate the percentages
    pivot_table_percentage = pivot_table.divide(pivot_table.sum(axis=1), axis=0) * 100
    # Reset index for Plotly express compatibility
    pivot_table_percentage = pivot_table_percentage.reset_index()
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(pivot_table_percentage, id_vars=['is_churned'], var_name=selected_column, value_name='Percentage')
    
    fig = px.bar(melted_df, x='is_churned', y='Percentage', color=selected_column,
                 labels={'Percentage': 'Percentage', 'is_churned': 'is_churned'},
                 title=f'Comparison of {selected_column} with Churn Status')
    fig.update_layout(barmode='stack')
    return fig

# Run the app

if __name__ == '__main__':
    app.run_server(debug=True)
