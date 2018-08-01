# Copyright 2017 Bo Shao. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import tensorflow as tf

from flask import Flask, request, jsonify
from settings import PROJECT_ROOT
from chatbot.botpredictor import BotPredictor
from flask import make_response

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
app = Flask(__name__)

print('rest service will be started.')
corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
sess = None
predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
                         result_dir=res_dir, result_file='basic')
	
	
@app.route("/")
def hello():
	print('hello will be called.')
	return "Hello!<br>\
	<br>\
	More AI services will be provided later.<br>\
	<br>\
	Available services are as below:<br>\
	<br>\
	1. gait recognition<br>\
	2. free chat engine<br>\
	<br>\
	<br>\
	<br>\
	Yours<br>\
	xfei.zhang<br>\
	"

@app.route('/reply', methods=['POST', 'GET'])
def reply():
	session_id = int(request.args.get('sessionId'))
	print('reply will be called.')
	question = request.args.get('question')
	print('session_id %s:', session_id)
	print('question %s:', question)
	if session_id not in predictor.session_data.session_dict:  # Including the case of 0
		session_id = predictor.session_data.add_session()
	answer = predictor.predict(session_id, question, html_format=True)
	response = make_response(answer)
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'POST'
	response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
	return response
	#return jsonify({'sessionId': session_id, 'sentence': answer})

#
# if __name__ == "__main__":
# 	print('main rest service will be started.')
# 	corp_dir = os.path.join(PROJECT_ROOT, 'Data', 'Corpus')
# 	knbs_dir = os.path.join(PROJECT_ROOT, 'Data', 'KnowledgeBase')
# 	res_dir = os.path.join(PROJECT_ROOT, 'Data', 'Result')
# 	with tf.Session() as sess:
# 		predictor = BotPredictor(sess, corpus_dir=corp_dir, knbase_dir=knbs_dir,
# 		                         result_dir=res_dir, result_file='basic')
# 	app.run()
# 	print("Web service started.")
