import json
import keras
from keras.models import load_model
import tensorflow as tf
from math import sqrt
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

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

print("Loading word model")
word_model = KeyedVectors.load_word2vec_format('./resources/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore') 
print("Word model loaded")

MAX_LENGTH_PARA = 300
MAX_LENGTH_COM = 300
blank = np.zeros(300)

global model
model = load_model("./resources/Guardian_gru_model.h5")
global graph
graph = tf.get_default_graph()

# All guardian articles now
def insert_comment_to_json(filepath, new_comment, para_id_list):
	try:
		data = json.load(open(filepath,"r"))
	except:
		data=[]
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


for i in range(4,22):
    article_path = "./static/data/dummy/Guardian/Article_"+ str(i)+".json"
    comment_path = "./static/data/dummy/Guardian/Comment_"+ str(i)+".txt"
    comment_json_path = "./static/data/dummy/Guardian/Comment_"+ str(i)+".json"

    comments = []
    comment_file = open(comment_path,"r")
    for line in comment_file:
        l = line.strip()
        l = line.strip("\n")

        comments.append(str(l))

    # print(comments)
    cnt = 0
    for c in comments:
        cnt += 1
        name = "user"+str(cnt)
        comment_vec = get_embedding(c,1)
        comment_vec = np.array([comment_vec])

        data = json.load(open(article_path,"r"))
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

        # create new comment
        new_comment = {}
        new_comment["id"] = cnt
        new_comment["comment"] = c
        new_comment["authorName"] = name
        new_comment["authorId"] = 100
        new_comment["authorAvatarUrl"] = "./static/img/jon_snow.png"
    
        insert_comment_to_json(comment_json_path, new_comment, para_id_list)
    
    # break