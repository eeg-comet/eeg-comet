

import pandas as pd
import plotly.express as px

features = pd.read_csv('C://Users//amin_//Documents//GitHub//output_test//test_study//extracted_features//extracted_features.csv')

# reshape the d dataframe suitable for statsmodels package
df_features = pd.melt(features.reset_index(), id_vars=['Filename'], value_vars=['FOC_A', 'FOC_B', 'FOC_C', 'FOC_D', 'FOC_E'])
# replace column names
df_features.columns = ['Filename', 'Feature', 'FOC']
df_features['FOC'] = df_features['FOC']

fig = px.box(df_features, x="Filename", y="FOC")
fig.show()