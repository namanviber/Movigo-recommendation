import pickle
import numpy as np
from flask import Flask,render_template,request,jsonify
import pandas as pd
from surprise.model_selection import train_test_split
from surprise.reader import Reader
from surprise import Dataset
from collections import defaultdict


app=Flask(_name_)
svd_model=pickle.load(open('Scaled_ratings.pkl','rb'))
svd_algo=pickle.load(open('svd_model.pkl','rb'))
df = pickle.load(open("testRating.pkl","rb"))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])

def predict():
    input_data=request.get_json()

    # input=np.array(input_data['inp'])
    # print(input_data)

    # user = similar_user(input_data)
    pred = user_recommendation(input_data,346)
    return pred

def user_recommendation(json_input, userid):

    # Json to Dataframe
    dfItem = pd.DataFrame.from_records(json_input)

    # Non Rated Movies Normalized Dataframe
    nonRated = df[~df['tmdbId'].isin(dfItem["tmdbId"])]
    user_ratings_mean = nonRated.groupby('userId')['rating'].mean()
    nonRated['rating'] = nonRated.apply(lambda row: row['rating'] - user_ratings_mean[row['userId']], axis=1)
    l1 = nonRated.rating.unique()

    # Building Testset
    reader = Reader(rating_scale=(l1.min(), l1.max()))
    data   = Dataset.load_from_df(nonRated[['userId','tmdbId','rating']], reader)
    train,test = train_test_split(data,test_size=0.99)

    # predtrainset = data.build_full_trainset()
    # predset = predtrainset.build_anti_testset()

    # Getting Recommendations
    testpred = svd_algo.test(test)

    # Returning 30 Top recommended Movies
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in testpred:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
            user_ratings.sort(key = lambda x: x[1], reverse = True)
            top_n[uid] = user_ratings[: 30 ]

    predicted_df = pd.DataFrame([(id, pair[0],pair[1]) for id, row in top_n.items() for pair in row],
                        columns=["userId" ,"tmdbId","rat_pred"])
    pred = predicted_df[predicted_df["userId"] == (userid)]["tmdbId"].tolist()

    return pred


if _name=="__main_":
    app.run(debug=True)