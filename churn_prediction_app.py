# Import required libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table

from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix

def calculate_model_metrics(model_name, X_test, y_test):
    model = joblib.load(model_name)
    predictions = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    # Remove the '.pkl' extension from the model name
    model_name_display = model_name.replace('.pkl', '')
    
    return {
        "Model": model_name_display,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Recall": round(recall_score(y_test, predictions), 4),
        "Precision": round(precision_score(y_test, predictions), 4),
        "F1": round(f1_score(y_test, predictions), 4),
        "AUC": round(roc_auc_score(y_test, y_score), 4)
    }

# Read the data once at the start of the app to avoid re-reading it in each callback
df = pd.read_csv('final_dataframe_churn_prediction.csv')

df['is_churned_label'] = df['is_churned'].replace({0: 'not churned', 1: 'churned'})

# Determining which are the numerical columns
categorical_cols = ['warehouse_Bogota', 'sub_categoria_Ambientadores', 'sub_categoria_Complementos y vitaminas', 'is_churned']
numerical_cols = [col for col in df.columns if col not in categorical_cols]

# Train Test Split
X = df.drop(columns=["is_churned", "is_churned_label"], axis=1).copy()
y = df['is_churned'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model_names = ['KNN.pkl', 'KNN_with_SMOTE.pkl', 'LogisticRegression.pkl', 'LogisticRegression_with_SMOTE.pkl',
               'RandomForest.pkl', 'RandomForest_with_SMOTE.pkl', 'XGBoost.pkl', 'XGBoost_with_SMOTE.pkl']

model_metrics = [calculate_model_metrics(model_name, X_test, y_test) for model_name in model_names]
metrics_df = pd.DataFrame(model_metrics)

def calculate_descriptive_statistics(df, numerical_cols):
    # this one helps to generate the table tha serves as a guidance for variable input in tab 2
    stats = df[numerical_cols].describe(percentiles=[.25, .5, .75])
    stats = stats.loc[['min', '25%', '50%', '75%', 'max']]
    stats.rename(index={'25%': 'Q1', '50%': 'median', '75%': 'Q3'}, inplace=True)
    stats = stats.round(4)
    stats_df = stats.T  # Transpose to have variables as rows
    stats_df.reset_index(inplace=True)  # Reset index to make the column names a regular column
    stats_df.rename(columns={'index': 'Variable'}, inplace=True)  # Rename the column for clarity
    return stats_df

def generate_input_boxes():
    # Calculate the descriptive statistics
    stats_df = calculate_descriptive_statistics(df, numerical_cols)

    # Function to find stats for a given column
    def find_stats(col):
        if col in stats_df['Variable'].values:
            row = stats_df[stats_df['Variable'] == col].iloc[0]
            return [html.Td(row[stat]) for stat in ['min', 'Q1', 'median', 'Q3', 'max']]
        return [html.Td('') for _ in range(5)]  # Empty cells if no stats

    # Combined rows for all variables
    combined_rows = []
    for col in X_train.columns:
        # Determine if the variable is categorical or numerical
        if col in categorical_cols:
            input_cell = html.Td(dcc.Dropdown(id=f'input-{col}', options=[{'label': v, 'value': v} for v in sorted(df[col].unique())]), style=styles['table_cell'])
        else:
            input_cell = html.Td(dcc.Input(id=f'input-{col}', type='number'), style=styles['table_cell'])

        # Add the row with input cell and stats cells
        combined_rows.append(html.Tr([html.Td(html.Label(col), style=styles['table_cell']), input_cell] + find_stats(col)))

    return html.Div([
        html.Table([
            html.Tr([
                html.Th('Variable', style=styles['table_header']),
                html.Th('Input', style=styles['table_header']),
                html.Th('Min', style=styles['table_header']),
                html.Th('Q1', style=styles['table_header']),
                html.Th('Median', style=styles['table_header']),
                html.Th('Q3', style=styles['table_header']),
                html.Th('Max', style=styles['table_header'])
            ]),
            *combined_rows
        ], style=styles['table']),
        html.Button('Predict', id='predict-button', n_clicks=0, style=styles['button']),
        html.Div(id='prediction-output', style=styles['prediction_output'])
    ])

# Initialize the Dash app with suppress_callback_exceptions set to True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Custom CSS styles
styles = {
    'main_container': {
        'fontFamily': 'Helvetica, sans-serif',
        'margin': 'auto',
        'width': '90%',
        'padding': '20px',
        'backgroundColor': '#F3F4F6',  # Light grey background
    },
    'header': {
        'textAlign': 'center',
        'color': '#2B3A67',  # Dark blue
        'marginBottom': '30px',
        'fontFamily': 'Helvetica, sans-serif',
        'fontSize': '30px'
    },
    'tab': {
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '10px 0',
        'backgroundColor': '#E1E1E1',  # Lighter grey for tabs
        'fontFamily': 'Helvetica, sans-serif',
        'fontSize': '18px',
        'color': '#4A4A4A'  # Dark grey text
    },
    'selected_tab': {
        # New style for active (selected) tab
        'borderRadius': '5px',
        'padding': '15px',
        'margin': '10px 0',
        'backgroundColor': '#D3D3D3',  # Slightly different background to indicate active state
        'fontFamily': 'Helvetica, sans-serif',
        'fontSize': '18px',
        'color': '#2B3A67',  # Different text color for active tab
        'fontWeight': 'bold'  # Optionally make the font bolder for active tab
    },
    'graph_container': {
        'display': 'inline-block',
        'width': '49%',
        'padding': '10px',
        'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'  # Adding a subtle shadow for depth
    },
    'dropdown': {
        'marginBottom': '10px',
        'borderColor': '#2B3A67'  # Consistent dark blue border
    },
    'checkbox': {
        'margin': '10px',
        'color': '#4A4A4A'  # Dark grey text for checkboxes
    },
    'button': {
        'backgroundColor': '#2B3A67',  # Dark blue background for buttons
        'color': 'white',  # White text on buttons
        'border': 'none',
        'padding': '10px 20px',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'margin': '10px 0'
    },
    'table': {
        'borderCollapse': 'collapse',
        'width': '100%',
        'marginTop': '20px'
    },
    'table_header': {
        'backgroundColor': '#4A90E2',  # Light blue header
        'color': 'white',  # White text in table headers
        'padding': '10px',
        'border': '1px solid #ddd'
    },
    'table_cell': {
        'border': '1px solid #ddd',  # Light grey borders
        'padding': '8px',
        'textAlign': 'left'
    },
    'h2_header': {
    'textAlign': 'center',
    'color': '#2B3A67',  # Same color as H1
    'marginBottom': '20px',  # Slightly smaller margin than H1
    'fontFamily': 'Helvetica, sans-serif',
    'fontSize': '24px'  # Smaller font size than H1
    },
    'data_table_header': {
    'backgroundColor': '#4A90E2',  # Use the same light blue header
    'color': 'white',
    'fontWeight': 'bold',
    'textAlign': 'center',
    'padding': '10px'
    },
    'data_table_cell': {
        'textAlign': 'left',
        'padding': '8px',
        'border': '1px solid #ddd'
    },
    'prediction_output': {
    'margin': '20px 0',
    'padding': '15px',
    'borderRadius': '5px',
    'backgroundColor': '#D3F2FF',  # Light blue background for emphasis
    'border': '2px solid #2B3A67',  # Dark blue border
    'color': '#2B3A67',  # Dark blue text for contrast
    'fontFamily': 'Helvetica, sans-serif',
    'fontSize': '18px',
    'textAlign': 'center',
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)',  # Subtle shadow for depth
    'fontWeight': 'bold'  # Bold text for emphasis
    }
}

# Define the layout of the app with additional styling
app.layout = html.Div([
    html.H1('Customer Classification Model: Churn Prediction', 
            style=styles['header']),
    
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='EDA', value='tab-1', style=styles['tab'], selected_style=styles['selected_tab']),
        dcc.Tab(label='Classifiers', value='tab-2', style=styles['tab'], selected_style=styles['selected_tab']),
    ]),
    
    html.Div(id='tabs-content')  # Add padding to the content area
], style=styles['main_container'])

# Callback for rendering tabs content
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        # Chart 1: Class count in a bar chart
        category_counts = df['is_churned_label'].value_counts()
        fig1 = px.bar(category_counts, x=category_counts.index, y=category_counts.values)
        fig1.update_layout(
            yaxis_title='',
            xaxis_title='',
            title={'text': 'Count of churned and non churned customers (Target Variable)', 'x': 0.5, 'xanchor': 'center'}
        )

        # Layout for the boxplot chart
        boxplot_layout = html.Div([
            html.H3("Numerical Variables", style=styles['h2_header']),
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
        
        # Bar chart for comparing other variables against the is_churned class
        bar_chart_layout = html.Div([
            html.H3("Categorical Variables", style=styles['h2_header']),
            dcc.Dropdown(
                id='comparison-dropdown',
                options=[{'label': col, 'value': col} for col in sorted(categorical_cols) if col not in ['is_churned_label', 'is_churned', 'customer_id']],
                value= [col for col in categorical_cols if col not in ['is_churned_label', 'is_churned', 'customer_id']][0]  # set the default value to the first column (excluding 'is_churned')
            ),
            html.Div([
                html.Div([dcc.Graph(figure=fig1)], style=styles['graph_container']),  # Graph for churn count
                html.Div([dcc.Graph(id='bar-chart')], style=styles['graph_container'])  # Graph for categorical comparison
            ], style={'display': 'flex'})  # This ensures the two graphs are side by side
        ])

        return html.Div([
            boxplot_layout,
            bar_chart_layout
        ])

    elif tab == 'tab-2':
        
        model_options = [
            {'label': 'KNN', 'value': 'KNN.pkl'},
            {'label': 'KNN with SMOTE', 'value': 'KNN_with_SMOTE.pkl'},
            {'label': 'Logistic Regression', 'value': 'LogisticRegression.pkl'},
            {'label': 'Logistic Regression with SMOTE', 'value': 'LogisticRegression_with_SMOTE.pkl'},
            {'label': 'Random Forest', 'value': 'RandomForest.pkl'},
            {'label': 'Random Forest with SMOTE', 'value': 'RandomForest_with_SMOTE.pkl'},
            {'label': 'XGBoost', 'value': 'XGBoost.pkl'},
            {'label': 'XGBoost with SMOTE', 'value': 'XGBoost_with_SMOTE.pkl'}
        ]
        return html.Div([
            dcc.Dropdown(
                id='model-dropdown',
                options=model_options,
                value='KNN.pkl',  # Default value
                style=styles['dropdown']
            ),
            html.Div([
                html.Div([dcc.Graph(id='confusion-matrix-graph')], style=styles['graph_container']),
                html.Div([dcc.Graph(id='roc-curve-graph')], style=styles['graph_container'])
            ], style={'display': 'flex', 'marginBottom': '20px'}),
            dash_table.DataTable(
                id='model-metrics-table',
                columns=[{"name": i, "id": i} for i in metrics_df.columns],
                data=metrics_df.to_dict('records'),
                style_cell=styles['data_table_cell'],
                style_header=styles['data_table_header']
            ),
            html.H2("Predict Module", style=styles['h2_header']),
            generate_input_boxes()
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
        
    fig = px.box(filtered_df, x=selected_column, y='is_churned_label')
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
    pivot_table = df.groupby(['is_churned_label', selected_column]).size().unstack().fillna(0)
    # Calculate the percentages
    pivot_table_percentage = pivot_table.divide(pivot_table.sum(axis=1), axis=0) * 100
    # Reset index for Plotly express compatibility
    pivot_table_percentage = pivot_table_percentage.reset_index()
    # Melt the DataFrame for easier plotting
    melted_df = pd.melt(pivot_table_percentage, id_vars=['is_churned_label'], var_name=selected_column, value_name='Percentage')
    
    fig = px.bar(melted_df, x='is_churned_label', y='Percentage', color=selected_column,
                 labels={'Percentage': 'Percentage', 'is_churned_label': 'is_churned'},
                 title=f'Comparison of {selected_column} with Churn Status')
    fig.update_layout(barmode='stack')
    return fig

def load_model_file(model_name):
    model = joblib.load(model_name)
    return model

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted Label", y="True Label", color="Number of Predictions"),
                    x=['not churned', 'churned'],
                    y=['not churned', 'churned'],
                    title="Confusion Matrix")
    # Set coloraxis_showscale to False to remove the color scale
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC = {auc_score:.2f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')

    return fig

@app.callback(
    [Output('confusion-matrix-graph', 'figure'),
     Output('roc-curve-graph', 'figure')],
    [Input('model-dropdown', 'value')]
)
def update_model_output(selected_model):
    # Load model and predict
    model = load_model_file(selected_model)
    predictions = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]  # Probability estimates for the positive class
    else:
        y_score = model.decision_function(X_test)

    # Confusion matrix and ROC curve figures
    confusion_matrix_fig = plot_confusion_matrix(y_test, predictions)
    roc_curve_fig = plot_roc_curve(y_test, y_score)

    return confusion_matrix_fig, roc_curve_fig

@app.callback(
    Output('model-metrics-table', 'style_data_conditional'),
    [Input('model-dropdown', 'value')]
)
def update_table_style(selected_model):
    selected_model_name = selected_model.replace('.pkl', '')  # Remove .pkl for matching
    return [{
        'if': {'filter_query': '{{Model}} = "{}"'.format(selected_model_name)},
        'backgroundColor': '#D2F3FF',  # or any other color you prefer
    }]

##################### predicts module ####################

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks'), Input('model-dropdown', 'value')],
    [State(f'input-{col}', 'value') for col in X_train.columns]
)
def predict(n_clicks, selected_model, *input_values):
    if n_clicks > 0:
        input_data = pd.DataFrame([input_values], columns=X_train.columns)
        model = load_model_file(selected_model)
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(input_data)
        return f'Prediction: {prediction[0]}, Probability: {proba[0]:.4f}'
    return ''

# Run the app

if __name__ == '__main__':
    app.run_server(debug=True)