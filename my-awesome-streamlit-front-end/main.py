# https://docs.streamlit.io/en/stable/troubleshooting/clean-install.html # install-streamlit-on-macos-linux

import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


header = st.container() # st.beta_container()
dataset = st.container() # st.beta_container()
features = st.container() # st.beta_container()
model_training = st.container() # st.beta_container()

st.markdown(
	"""
	<style>
	.main{
		background: #f5f5f5;
	}
	</style>
	""",
	unsafe_allow_html=True
)

@st.cache 
def get_data(filename):
	taxi_data = pd.read_csv(filename)

	return taxi_data


with header: 
	st.title("Welcome to my awesome data science project!")
	st.text('In this project I look into the transactions of taxis in NYC.')


with dataset: 
	st.header("NYC taxi dataset")
	st.text('I found this dataset on blablabla.com, ...')

	taxi_data = get_data("data/taxi_data.csv")
	st.write(taxi_data.head()) #1:29

	st.subheader('Pick-up location ID distribution on the NYC dataset') # add a title to bar_chart
	pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
	st.bar_chart(pulocation_dist)


with features: 
	st.header("The features I created")
	#st.text('In this project I look into the transactions of taxis in NYC.')

	st.markdown("* **first feature:** I created this feature because of this... I calculated it using this logic...")
	st.markdown("* **second feature:** I created this feature because of this... I calculated it using this logic...")


with model_training: 
	st.header("Time to train the model!")
	st.text('Here you get to choose the hyperparameters of the model an see how the performance changes!')

	selection_col, display_col = st.columns(2) #st.beta_columns(2) # 2 is the number of columns

	max_depth = selection_col.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=10)

	n_estimators = selection_col.selectbox("How many trees shoud there be?", options=[100,200,300, 'No limit'], index=0)

	#selection_col.text('Here is a list of features in my data:') # professor's code
	#selection_col.write(taxi_data.columns) # or st.table(taxi_data.columns) # professor's code
	#input_feature = selection_col.text_input("Which feature shoud be used as the input feature?", "PULocationID") # professor's code

	columnsList = list(taxi_data.columns)
	# columnsList # see index columns
	columnsList[0], columnsList[16] = columnsList[16], columnsList[0] # choose total_amount to be the one by default as input feature
	input_feature = selection_col.selectbox("Which feature shoud be used as the input feature?", columnsList) #


	if n_estimators == 'No limit':
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

	X = taxi_data[[input_feature]]
	y = taxi_data[['trip_distance']]

	regr.fit(X, y)
	prediction = regr.predict(y)

	display_col.subheader("Mean absolute error of the model is:")
	display_col.write(mean_absolute_error(y, prediction))

	display_col.subheader("Mean squared error of the model is:")
	display_col.write(mean_squared_error(y, prediction))

	display_col.subheader("R squared error of the model is:")
	display_col.write(r2_score(y, prediction))


# to create automatically the requierementes.txt
# $ pip install pipreqs
# $ pipreqs ./
