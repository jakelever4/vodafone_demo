import pandas as pd
import json
from datetime import datetime, timedelta
import statistics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn import metrics
from dateutil import parser


# Import data
pd.set_option("display.max_rows", None, "display.max_columns", None)
fires_df = pd.read_csv('fires_df.csv')

# Convert rows
fires_df['sentiment'] = fires_df['sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_positive_sentiment'] = fires_df['overall_positive_sentiment'].apply(lambda x: json.loads(x))
fires_df['overall_negative_sentiment'] = fires_df['overall_negative_sentiment'].apply(lambda x: json.loads(x))
fires_df['magnitude'] = fires_df['magnitude'].apply(lambda x: json.loads(x))
fires_df['num_tweets'] = fires_df['num_tweets'].apply(lambda x: json.loads(x))
fires_df['avg_sentiment'] = fires_df['avg_sentiment'].apply(lambda x: json.loads(x))
fires_df['start_doy'] = fires_df['start_date'].apply(lambda  x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)
fires_df['end_doy'] = fires_df['end_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').timetuple().tm_yday)

avg_magnitude = []
for ind, row in fires_df.iterrows():
    m = [x / y for x,y in zip(row['magnitude'],row['num_tweets'])]
    avg_magnitude.append(m)

fires_df['avg_magnitude'] = avg_magnitude
fires_df['s_mean'] = fires_df['sentiment'].apply(lambda x: statistics.mean(x))
fires_df['s_var'] = fires_df['sentiment'].apply(lambda x: np.var(x))
fires_df['m_mean'] = fires_df['magnitude'].apply(lambda x: statistics.mean(x))
fires_df['m_var'] = fires_df['magnitude'].apply(lambda x: np.var(x))
fires_df.drop(columns=['duration'])

# categorical variables to convert
# print(fires_df['direction'].value_counts())
# print(fires_df['landcover'].value_counts())
# print(fires_df['state'].value_counts())

fires_df['direction_cat'] = fires_df['direction'].astype('category').cat.codes
fires_df['landcover_cat'] = fires_df['landcover'].astype('category').cat.codes
fires_df['state_cat'] = fires_df['state'].astype('category').cat.codes


def get_unique_days_for_fire(start_date, duration):
    start_date = parser.parse(start_date)
    # start_date = datetime.strptime(start_date, '%Y %M %d')
    date_list = [start_date - timedelta(days=x) for x in range(duration)]
    return date_list


def plot_sentiment_for_fire(fire):
    print('Creating sentiment vector graph for fire_ID: {}'.format(fire['fire_ID']))
    dates = get_unique_days_for_fire(fire['start_date'], fire['s_duration'])
    short_dates = []
    for date in dates:
        short_dates.append(date.strftime('%Y-%m-%d'))
    dates = short_dates

    fig, axs = plt.subplots(2, 2, sharex=True, figsize=(14, 8))
    fig.suptitle('fire ID: {}, Location: {}. over burn period ({} to {})'.format(fire['fire_ID'], fire['location'], fire['start_date'], fire['end_date']))

    axs[0,0].set_title('SUM(Sentiment)')
    axs[0,0].set(xlabel='Date', ylabel='SUM(Sentiment)')
    axs[0,1].set_title('SUM(Magnitude)')
    axs[0,1].set(ylabel='SUM(Magnitude)')
    axs[1,0].set(ylabel='AVG(Sentiment)')
    axs[1,0].set_title('AVG(Sentiment)')
    axs[1,1].set(ylabel='AVG(Magnitude)')
    axs[1,1].set_title('AVG(Magnitude)')

    magnitude = fire['magnitude']
    sentiment = fire['sentiment']
    num_tweets = fire['num_tweets']
    avg_sentiment = fire['avg_sentiment']
    avg_magnitude = fire['avg_magnitude']
    pos_sentiment = fire['overall_positive_sentiment']
    neg_sentiment = fire['overall_negative_sentiment']

    axs[0,0].plot(dates, sentiment, color='b')
    axs[0,0].bar(dates, pos_sentiment, color='g')
    axs[0,0].bar(dates, neg_sentiment, color='r')
    axs[0,1].plot(dates, magnitude, color='orange')
    axs[1,0].plot(dates, avg_sentiment, color= 'cyan') #
    axs[1,0].plot(dates, np.zeros(len(magnitude)), color='deepskyblue', ls=':')
    axs[1,1].plot(dates, avg_magnitude, color='purple')

    axs[1,0].set_ylim([-1, 1])

    plt.setp(axs[1,1].get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(axs[1,0].get_xticklabels(), rotation=30, horizontalalignment='right')

    # plt.show()
    fig.savefig('graphs/sentiment_vectors/{}.png'.format(fire['fire_ID']))

    plt.close(fig)
    return None


def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def predict_cat(X, target_var, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, target_var, test_size=0.25)

    # Various hyper-parameters to tune
    xgb1 = xgb.XGBClassifier()
    parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
                  'objective': ['binary:logistic'], #['reg:linear'],
                  'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], #so called `eta` value
                  'max_depth': [7, 8, 9, 10],
                  'min_child_weight': [4],
                  'verbosity': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.4, 0.6],
                  'n_estimators': [100, 200, 300, 400, 500]
                  }

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 4,
                            n_jobs = 4,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    print('Best parameters from grid search: ')
    print(xgb_grid.best_params_)

    y_pred = xgb_grid.predict(X_test)
    y_test = y_test.to_numpy()
    print('Test set results:')
    for i in range(len(y_pred)):
        print('True: {} pred: {}'.format(y_test[i], y_pred[i]))

    print('Results for target variable: {}'.format(target_name))

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('accuracy: {}'.format(accuracy))

    xgb.plot_tree(xgb_grid.best_estimator_,num_trees=0)
    plt.savefig('graphs/trees/{}.png'.format(target_name), dpi=2000, bbox_inches='tight')
    # plt.show()

    # here the f score is how often the variable is split on - i.e. the F(REQUENCY) score
    xgb.plot_importance(xgb_grid.best_estimator_)
    plt.tight_layout()
    plt.savefig('graphs/feature_importances/{}.png'.format(target_name))
    # plt.show()
    plt.close()

    return accuracy


def predict(X, target_var, target_name):
    X_train, X_test, y_train, y_test = train_test_split(X, target_var, test_size=0.25)

    # Various hyper-parameters to tune
    xgb1 = xgb.XGBRegressor()
    parameters = {'nthread': [4], #when use hyperthread, xgboost may become slower
                  'objective': ['reg:linear'], #['binary:logistic'],
                  'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], #so called `eta` value
                  'max_depth': [7, 8, 9, 10],
                  'min_child_weight': [4],
                  'verbosity': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.4, 0.6],
                  'n_estimators': [100, 200, 300, 400, 500]
                  }

    xgb_grid = GridSearchCV(xgb1,
                            parameters,
                            cv = 4,
                            n_jobs = 4,
                            verbose=True)

    xgb_grid.fit(X_train, y_train)

    # print(xgb_grid.best_score_)
    print('Best parameters from grid search: ')
    print(xgb_grid.best_params_)

    y_pred = xgb_grid.predict(X_test)
    y_test = y_test.to_numpy()

    print('Test set results:')
    for i in range(len(y_pred)):
        print('True: {} pred: {}'.format(y_test[i], y_pred[i]))

    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)

    gini_predictions = gini(y_test, y_pred)
    gini_max = gini(y_test, y_test)
    ngini= gini_normalized(y_test, y_pred)
    print('Gini: %.3f, Max. Gini: %.3f, Normalized Gini: %.3f' % (gini_predictions, gini_max, ngini))

    print('Results for target variable: {}'.format(target_name))

    print('R2 Score: {}'.format(r2))
    print('MAE: {}'.format(mae))
    print('MSE: {}'.format(mse))

    xgb.plot_tree(xgb_grid.best_estimator_,num_trees=0)
    plt.savefig('graphs/trees/{}.png'.format(target_name), dpi=2000, bbox_inches='tight')
    # plt.show()

    # here the f score is how often the variable is split on - i.e. the F(REQUENCY) score
    xgb.plot_importance(xgb_grid.best_estimator_)
    plt.tight_layout()
    plt.savefig('graphs/feature_importances/{}.png'.format(target_name))
    # plt.show()
    plt.close()

    return r2, mae, mse


# PART 1 F(X) - PREDICTING SOCIAL SENTIMENT VALUES
predictors = ['latitude', 'longitude', 'size', 'perimeter', 's_duration', 'speed', 'expansion', 'pop_density', 'direction_cat', 'landcover_cat', 'state_cat', 'start_doy', 'end_doy']
targets = ['s_mean', 'm_mean', 'overall_magnitude', 'overall_sentiment', 'total_tweets', 's_var', 'm_var']
X = fires_df[predictors]
X2 = fires_df[targets]

# run predictions for each of the target variables
r2s = []
maes = []
mses = []
for target in targets:
    r2, mae, mse = predict(X, fires_df[target], target)
    r2s.append(r2)
    maes.append(mae)
    mses.append(mse)

# show results from prediction
print("RESULTS")
for i in range(len(targets)):
    print('Variable: {}. Results: R2 Score: {}, MAE: {}. MSE: {}'.format(targets[i], r2s[i], maes[i], mses[i]))


# PART 2 G(X) - PREDICTING PHYSICAL WILDFIRE VARIABLES
targets = []
cat_targets = []
r2s = []
maes = []
mses = []
accs = []
for pred in predictors:
    if pred == 'landcover_cat' or pred == 'state_cat' or pred == 'direction_cat':
        acc = predict_cat(X2, fires_df[pred], pred)
        cat_targets.append(pred)
        accs.append(acc)
    else:
        r2, mae, mse = predict(X2, fires_df[pred], pred)
        targets.append(pred)
        r2s.append(r2)
        maes.append(mae)
        mses.append(mse)

# show results from prediction
print("RESULTS")
for i in range(len(targets)):
    print('Variable: {}. Results: R2 Score: {}, MAE: {}. MSE: {}'.format(targets[i], r2s[i], maes[i], mses[i]))

for i in range(len(cat_targets)):
    print('Variable: {}. Results: Accuracy: {}'.format(cat_targets[i], accs[i]))


# Part 3 - Plot & save Sentimental Curves of Wildfires
for ind, row in fires_df.iterrows():
    plot_sentiment_for_fire(row)