import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("var.csv")
df = df.dropna()

X = df[["Patents_Granted", "Trademarks_Registered", "Duration_between_founded_to_reaching_a_unicorn_list_in_years","Duration_between_unicorn_listed_date_and_exit_date_in_years","Duration_between_first_funding_round_and_last_funding_round_in_years"]]


y = df["Status_target_variable"]

from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X,y, test_size = 0.2, random_state = 10)

#Shows the shape of training data set
print(X_train.shape,y_train.shape)


#Shows the shape of testing data set
print(X_cv.shape,y_cv.shape)


clf = LogisticRegression() 
clf.fit(X_train, y_train)



from sklearn.metrics import accuracy_score
pred_cv = clf.predict(X_cv)
accuracy_score(y_cv,pred_cv)
print("The testing accuracy:",accuracy_score)

pred_train = clf.predict(X_train)
accuracy_score(y_train,pred_train)
print("The training accuracy:",accuracy_score)

joblib.dump(clf, "clf.pkl")