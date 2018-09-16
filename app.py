# encoding=utf8  

import sys
import os


from flask import Flask
from flask import Flask, render_template, url_for, request, session, redirect
from flask_session import Session
import json
import os
from keras.models import load_model
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from math import sqrt

# import h5py
# from keras.models import Model
# from keras.layers import Bidirectional, GRU, Reshape, RepeatVector, dot, Dot, Input, LSTM, merge, Dense, TimeDistributed, Dropout, Embedding, Activation, Reshape, Lambda, Flatten, Add, Multiply, Concatenate
# from keras import optimizers


# from keras.activations import softmax
import tensorflow as tf
'''
import pickle
from sklearn.model_selection import StratifiedKFold
import keras

'''
app = Flask(__name__)
sess = Session()

print("Loading word model")
word_model = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore') 
print("Word model loaded")

global model
model = load_model("./resources/Guardian_gru_model.h5")
global graph
graph = tf.get_default_graph()

MAX_LENGTH_PARA = 300
MAX_LENGTH_COM = 300
blank = np.zeros(300)
'''
#Model
para_input = Input(shape=(MAX_LENGTH_PARA,300,), name='para_input')
comment_input = Input(shape=(MAX_LENGTH_COM,300,), name='comment_input')
bigru_para = Bidirectional(GRU(units = 150, return_sequences=False), name="bigru_para")(para_input)
bigru_com = Bidirectional(GRU(units = 150, return_sequences=False), name="bigru_com")(comment_input)
merge = Concatenate()([bigru_para, bigru_com])
# l_dense = TimeDistributed(Dense(1000, activation='tanh'))(merge)

class_output = Dense(units=5, activation="softmax", name="class_output")(merge)

model = Model(inputs=[para_input, comment_input], outputs=[class_output])
rmsprop = optimizers.RMSprop(lr=0.001, decay=1e-6)#, momentum=0.9, nesterov=True)
model.compile(optimizer=rmsprop, loss="categorical_crossentropy", metrics=["accuracy"])

model.load_weights("./resources/Guardian_gru_weights.h5")
'''



def get_embedding(text,flag=0):
	l = text.split()
	encoding = []
	if flag==0:
		MAX_LENGTH = MAX_LENGTH_PARA
	else:
		MAX_LENGTH = MAX_LENGTH_COM

	if len(l) < MAX_LENGTH :
		for i in range(MAX_LENGTH - len(l)):
			encoding.append(blank)

	n = min(MAX_LENGTH,len(l))
	for i in range(len(l)) : 
		t = l[i]
		try : 
			enc = word_model[t]
		except :
			enc = blank
		encoding.append(enc)
	encoding = np.asarray(encoding)
	return encoding

def get_mean(vec):
	length = len(vec)
	e = np.zeros(300)
	for i in range(length):
		for j in range(300):
			e[j] += vec[i][j]
	for j in range(300):
		e[j] /= length
	return e    

def get_input_vec(v1, v2):
	sum = 0
	e = np.zeros(300)
	for i in range(300):
		e[i] = v1[i] * v2[i]
		sum += (e[i]*e[i]) 
	sum = sqrt(sum)
	for i in range(300):
		e[i] /= sum


# @app.context_processor
# def override_url_for():
#     return dict(url_for=dated_url_for)

# def dated_url_for(endpoint, **values):
#     if endpoint == 'static':
#         filename = values.get('filename', None)
#         if filename:
#             file_path = os.path.join(app.root_path,
#                                  endpoint, filename)
#             values['q'] = int(os.stat(file_path).st_mtime)
#     return url_for(endpoint, **values)


def load_article(filepath):
	data = json.load(open(filepath,"r"))
	x = []
	p = []
	headline = ""
	for d in data:
		if str(d["sectionId"]) != "-1":
			x.append(str(d["sectionId"]))
			p.append(str(d["text"]))
		else:
			headline = str(d["text"])

	return x, p, headline

def load_comments_for_js(filepath):
	data = json.load(open(filepath,"r"))
	# print(data)
	return json.dumps(data)

def load_comments(filepath):
	data = json.load(open(filepath,"r"))
	ids = set()
	text_list = []
	name_list = []
	for d in data:
		try : 
			section_comment_list = d["comments"]
			for c in section_comment_list:
				id = int(c["id"])
				if id not in ids: 
					name_list.append(str(c["authorName"]))
					text_list.append(str(c["comment"]))
					ids.add(id)
	
		except:
			pass
	return text_list, name_list

@app.route("/")
def main():
	return render_template('index.html')

@app.route("/<variable1>/Article_<variable2>")
def article(variable1, variable2):
	article_json_path = "./static/data/dummy/" + str(variable1) + "/Article_" + str(variable2)+".json"
	comment_json_path = "./static/data/dummy/" + str(variable1) + "/Comment_" + str(variable2)+".json"
	
	# load paragraph json
	id_list, para_text_list, headline = load_article(article_json_path)
	comments = load_comments_for_js(comment_json_path)
	
	comment_text_list, commentor_name_list = load_comments(comment_json_path)
	# id_list = ["4","5"]
	# para_text_list = ["asdas","|wqewqeweq"]
	return render_template('article.html',paper = variable1,headline=headline, commentPass=zip(comment_text_list,commentor_name_list), toPass=zip(id_list,para_text_list),  all_comment_text=comments)






def num_comments(filepath):
	data = json.load(open(filepath,"r"))
	ids = set()
	for d in data:
		try:
			for c in d["comments"]:
				ids.add(int(c["id"]))
		except:
			pass
	
	return len(ids)

def insert_comment_to_json(filepath, new_comment, para_id_list):
	data = json.load(open(filepath,"r"))
	
	for pid in para_id_list:
		found = 0
		for idx, item in enumerate(data):
			if item["sectionId"] == para_id_list:
				data[idx]["comments"].append(new_comment)
				found = 1
				break
		if found == 0:
			nd = {}
			nd["sectionId"] = pid
			nd["comments"] = []
			nd["comments"].append(new_comment)

			data.append(nd)

	with open(filepath, 'w') as fp:
		json.dump(data, fp)


@app.route("/<variable1>/Article_<variable>", methods=['POST'])
def new_comment(variable1, variable2):
	article_json_path = "./static/data/dummy/" + str(variable1) + "/Article_" + str(variable2)+".json"
	comment_json_path = "./static/data/dummy/" + str(variable1) + "/Comment_" + str(variable2)+".json"
	
	name = str(request.form['name'])
	comment = str(request.form['message'])

	print (name)
	print (comment)


	
	comment_vec = get_embedding(comment,1)
	comment_vec = np.array([comment_vec])

	data = json.load(open(article_json_path,"r"))
	para_id_list = []
	for d in data:
		x = str(d["sectionId"])
		p = str(d["text"])
		para_vec = get_embedding(p,0)
		para_vec = np.array([para_vec])

		with graph.as_default():
			scores = model.predict([para_vec, comment_vec])
		
		scores = np.squeeze(scores, axis=0)
		rel_score = np.argmax(scores)

		print(rel_score)
		if rel_score == 3 or rel_score == 4:
			para_id_list.append(x)

	if len(para_id_list) == 0:
		para_id_list.append("-1")

	

	## add to comments json

	# find number of comments 
	comment_cnt = num_comments(comment_json_path)
	
	# create new comment
	new_comment = {}
	new_comment["id"] = comment_cnt+1
	new_comment["comment"] = comment
	new_comment["authorName"] = name
	new_comment["authorId"] = 100
	new_comment["authorAvatarUrl"] = "./static/img/jon_snow.png"

	# find position and insert
	insert_comment_to_json(comment_json_path, new_comment, para_id_list)


	## render back the page
	# load paragraph json
	id_list, para_text_list, headline = load_article(article_json_path)
	comments = load_comments_for_js(comment_json_path)
	
	comment_text_list, commentor_name_list = load_comments(comment_json_path)
	# id_list = ["4","5"]
	# para_text_list = ["asdas","|wqewqeweq"]
	return render_template('article.html',headline=headline, commentPass=zip(comment_text_list,commentor_name_list), toPass=zip(id_list,para_text_list),  all_comment_text=comments)


@app.route("/post")
def post():
	# load paragraph json
	id_list, para_text_list = load_article("./static/data/article/test_article.json")
	comments = load_comments("./static/data/comments/test_comments.json")
	# id_list = ["4","5"]
	# para_text_list = ["asdas","|wqewqeweq"]
	return render_template('post.html', toPass=zip(id_list,para_text_list),  all_comment_text=comments)


@app.route("/index")
def index():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def my_form_post():
	text = request.form['text']
	processed_text = str(text)
	processed_text = processed_text.strip()
	print(processed_text)

	# load the model here
	
	# predict score for all paragraphs using the model

	
	# get the paragraph with the highest score

	# append the comment for that section

	id_list, para_text_list = load_article("./static/data/article/test_article.json")
	comments = load_comments("./static/data/comments/test_comments.json")
	return render_template('index.html', toPass=zip(id_list,para_text_list),  all_comment_text=comments)

if __name__ == "__main__":
	app.secret_key = 'nips_cnerg'
	app.config['SESSION_TYPE'] = 'filesystem'
	sess.init_app(app)
	app.debug = False
	app.run()
