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


		
	
if __name__ == "__main__":
    app.run()
