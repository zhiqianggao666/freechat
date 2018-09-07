from rasa_nlu.model import Interpreter
import json
interpreter = Interpreter.load("./models/current/nlu")
message = "我正在寻找华为旁边的韩国料理食物"
result = interpreter.parse(message)
print(json.dumps(result, indent=2))
print(result.values())


