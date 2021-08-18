The script main.py runs a condensed version of my analysis using some test data that is supplied. 
The data is in fires_df.csv and contains entries of different socially analysed wildfire events from the global fire atlas 2016 ignitions dataset. 
I have downloaded and sentimentally analysed large number of tweets surrounding wildfire events, and created this dataset of wildfire events and their calculated online social sentiment values. 
Therefore, the data contains both physical and social variables for wildfire events.
Upon running the script, we load in the data, convert/preprocess it, and then run three main functions:
1. Predict social sentiment variables using only physical wildfire characteristics. The script fits a gradient boosted decision tree to predict social sentiment variables baed only on the phyasical data.
2. Predict physical variables from social sentiment values. The script uses similar ML methods to attempt to predict physcial aspects of wildfire activity based only on social sentiment values
3. Plot/save wildfire sentimental arcs. For each of the social sentiment variables, create a sentimental arc for the burn period of the wildfire & plot/save.

Results from the predictions are printed to the console, and the graphical results are stored in the graphs folder. There are three folders:
- feature importances : this folder shows a bar chart for the feature importances from each of the models. The model target is the graph filename.
- trees : example trees from each of the models. The model target is the graph filename.
- Sentiment vectors : each wildfire event has its own sentimental arcs which are saved here. Filename corresponds to fire_ID from fires_df.csv

python package requirements
- pandas
- matplotlib
- sklearn
- xgboost
- graphviz

system requirements
- about 650MB disk space
- 8GB system memory
