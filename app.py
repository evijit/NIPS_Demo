# encoding=utf8  

import sys
import os


from flask import Flask
from flask import Flask, render_template, url_for, request, session, redirect
from flask_session import Session
import json
import os
app = Flask(__name__)
sess = Session()


def load_article(filepath):
	data = json.load(open(filepath,"r"))
	x = []
	p = []
	for d in data:
		x.append(str(d["sectionId"]))
		p.append(str(d["text"]))

	return x, p

def load_comments(filepath):
	data = json.load(open(filepath,"r"))
	print(data)
	return json.dumps(data)

@app.route("/")
def main():
	# load paragraph json
	id_list, para_text_list = load_article("./static/data/article/test_article.json")
	comments = load_comments("./static/data/comments/test_comments.json")
	# id_list = ["4","5"]
	# para_text_list = ["asdas","|wqewqeweq"]
	return render_template('index.html', toPass=zip(id_list,para_text_list),  all_comment_text=comments)


# @app.route("/index")
# def index():
# 	return render_template('index.html')

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
	app.debug = True
	app.run()
