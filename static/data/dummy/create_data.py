import csv
import pandas as pd
import re
import pickle
import json

csvfilename='labelled_datas.tsv'
df = pd.read_csv(csvfilename,delimiter='\t',encoding='utf-8')
print(list(df.columns.values))

article_id_dict={}
old_comment=''
first_iter=False

for i,d in df.iterrows():
	try:
		comment= d['comment_name']
		para= d['paragraph_name']
		art_id= d['article_id_story']

		if comment==old_comment and first_iter==True:
			sec_id+=1
			article_id_dict[art_id][0][sec_id]=para

		if art_id not in article_id_dict:
			article_id_dict[art_id]=[{},[]]
			sec_id=1
			article_id_dict[art_id][0][sec_id]=para
			article_id_dict[art_id][1].append(comment)
			first_iter=True
			old_comment=comment

		if comment!= old_comment:
			article_id_dict[art_id][1].append(comment)
			old_comment=comment
			first_iter= False


	except Exception as e:
		print(d)
		continue


c=4

for i in article_id_dict:

	json_file_name= 'Article_'+str(c)+'.json'
	arr=[]

	for k in article_id_dict[i][0]:
		a={}
		a['sectionId']=str(k)
		a['text']=article_id_dict[i][0][k]
		arr.append(a)

	with open(json_file_name,'w') as outfile:
		json.dump(arr,outfile, indent=4)			

	comment_file=open('Comment_'+ str(c)+'.txt','w')	

	for k in article_id_dict[i][1]:
		comment_file.write(k+'\n')

	c+=1
		
	# print('Article ID')
	# print(i)

	# print('Paragraphs')
	# for k in article_id_dict[i][0]:
	# 	print(k, article_id_dict[i][0][k])

	# print('Comments')
	# for k in article_id_dict[i][1]:
	# 	print(k)

	# print('\n')	

		