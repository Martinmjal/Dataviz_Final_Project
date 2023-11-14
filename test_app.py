import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Read the dataframe
df = pd.read_csv('churn_prediction_features.csv')

df = df[['order_count_3m', 'order_count_historic', 'distinct_sku_3m', 'distinct_sku_historic',
     'gmv_3m', 'gmv_historic', 'unique_vendors_last_3m', 'unique_vendors_historic', 'AOV_3m', 'AOV_historic',
     'antiquity_days', 'visits_count_3m', 'canceled_orders', 'tickets_finales', 'has_churned_before', 'mean_order_lapse',
     'KVI_gmv_percent', 'FORE_gmv_percent', 'LONG_gmv_percent', 'NO_CLASS_gmv_percent', 'new_skus_within_3m', 'is_churned',
     'orders_count_delta_1m', 'gmv_delta_1m', 'distinct_skus_delta_1m', 'total_discount_delta_1m', 'orders_count_delta_2m',
     'gmv_delta_2m', 'distinct_skus_delta_2m', 'total_discount_delta_2m', 'orders_count_delta_3m', 'gmv_delta_3m', 'distinct_skus_delta_3m',
     'total_discount_delta_3m']]

# Assuming 'is_churned' is also a numerical binary column
# Extract numerical columns, excluding 'is_churned' and other non-relevant columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
numerical_cols = [col for col in numerical_cols if col not in ['has_churned_before', 'is_churned']]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Churn Prediction Feature Analysis"),
    dcc.Dropdown(
        id='numerical-dropdown',
        options=[{'label': col, 'value': col} for col in numerical_cols],
        value=numerical_cols[0]  # set the default value to the first numerical column
    ),
    dcc.Graph(id='boxplot-graph'),
])

# Define the callback to update the graph
@app.callback(
    Output('boxplot-graph', 'figure'),
    [Input('numerical-dropdown', 'value')]
)
def update_graph(selected_column):
    # Create the boxplot figure using the selected numerical column
    fig = px.box(df, x=selected_column, y="is_churned", notched=True)
    fig.update_traces(orientation='h')  # make boxplot horizontal
    fig.update_layout(title=f'Boxplot of {selected_column} by is_churned')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
