import dash
import seaborn as sns
import matplotlib.pyplot as plt
from dash import dcc, html
import base64
from io import BytesIO

# Sample data
tips = sns.load_dataset("tips")

# Create a seaborn figure
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x="total_bill", y="tip", data=tips)

# Save the plot to a BytesIO object
img_bytes = BytesIO()
plt.savefig(img_bytes, format='PNG')
plt.close()
img_bytes.seek(0)
base64_string = base64.b64encode(img_bytes.read()).decode()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(base64_string))
])

if __name__ == '__main__':
    app.run_server(debug=True)
