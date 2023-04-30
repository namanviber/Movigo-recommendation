import pickle
import numpy as np
from flask import Flask,render_template,request,jsonify
import pandas as pd
from surprise.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from surprise.reader import Reader
from surprise import Dataset
from collections import defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity

app=Flask(__name__)
user_ratings_scaled=pickle.load(open('Scaled_ratings.pkl','rb'))
user_ratings = pickle.load(open('pivot.pkl','rb'))
svd_algo=pickle.load(open('newModel.pkl','rb'))
df = pickle.load(open("testRating.pkl","rb"))
df = pd.read_csv('testRatings.csv')


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])

def predict():
    input_data=request.get_json()

    user = similar_user(input_data)
    pred = user_recommendation(input_data,user)
    # item = item_recommendation(input_data)
    # dict = {"user":pred,"item":item}
    # return dict
    return pred

def similar_user(json_input):

    # Json to Dataframe
    dfItem = pd.DataFrame.from_records(json_input)

    # Converting to pivot
    user = dfItem.pivot_table(columns='tmdbId', values='rating')
    concat = pd.concat([user_ratings,user], ignore_index= True)
    newuser = concat.tail(1)
    newuser.fillna(0, inplace=True)

    # Scale pivot element
    arr = newuser.to_numpy()
    scaler = MinMaxScaler()
    scaler.fit_transform(user_ratings_scaled)
    scaled_arr = scaler.transform(arr.reshape(1,-1))

    # getting similar user
    similarity_scores = cosine_similarity(scaled_arr, user_ratings_scaled)

    # print similar user
    similarity_scores = list(enumerate(similarity_scores[0]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_k_users = [i[0] for i in similarity_scores[1:10+1]]
    return top_k_users[0]

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
    dict = {"tmdbId":pred}
    output = json.dumps(dict, indent=2)

    return pred


if __name__=="__main__":
    app.run( debug= True)
   