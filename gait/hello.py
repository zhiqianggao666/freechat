from singlepredicting import test_predict
from flask import Flask
from flask import request



app = Flask(__name__)

@app.route("/")
def hello():
	return "Hello!<br>\
	<br>\
	More AI services will be provided later.<br>\
	<br>\
	Available services are as below:<br>\
	<br>\
	1. gait recognition<br>\
	<br>\
	<br>\
	<br>\
	Yours<br>\
	xfei.zhang<br>\
	"

@app.route("/gait")
def gait():
	return "Hello!<br>\
	<br>\
	Gait serverice is prepraring , plz be patient.<br>\
	<br>\
	Available sub-services are as below:<br>\
	<br>\
	gait/list_models<br>\
	<br>\
	gait/get_mapped_data<br>\
	<br>\
	gait/predict<br>\
	<br>\
	<br>\
	<br>\
	Yours<br>\
	xfei.zhang<br>\
	"
	
@app.route("/gait/list_models")
def gait_list_models():
	return "We've invited eight members to record their gait.<br>\
	<br>\
	Their name and index are as below:<br>\
	<br>\
	1:ts<br>\
	2:wsq<br>\
	3:wsw<br>\
	4:xyj<br>\
	5:lzq<br>\
	6:zyj<br>\
	7:ykr<br>\
	8:zxf<br>\
	"

@app.route("/gait/get_mapped_data")
def gait_get_mapped_data():
	return "1:ts|2:wsq|3:wsw|4:xyj|5:lzq|6:zyj|7:ykr|8:zxf"

	
@app.route("/gait/predict",methods = ['POST'])
def gait_predict():
	filename = 'test.csv'
	if request.headers['Content-Type'] == 'application/octet-stream':
		f = open(filename, 'wb')
		f.write(request.data)
		f.close()
		predicted_index,acc = test_predict(filename)
		return str(predicted_index)+":"+str(acc)
		
	
if __name__ == "__main__":
    app.run()
