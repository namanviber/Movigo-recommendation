import pickle
import numpy as np
from flask import Flask,render_template,request,jsonify
import os

app=Flask(__name__)
svd_model=pickle.load(open('Scaled_ratings.pkl','rb'))



@app.route('/predict',methods=['POST','GET'])
def predict():
    input_data=request.get_json()

    # input=np.array(input_data['inp'])
    # print(input_data)
    user_ratings = [] 
    for rating in input_data:
        user_id = rating['userid']
        movie_id = rating['tmdbid']
        rating_value = rating['rating']
        user_ratings.append((user_id, movie_id, rating_value))

    print(user_ratings)
    return ''


port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)








    

    

    

    