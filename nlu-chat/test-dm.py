#from rasa_core.agent import Agent
# agent = Agent.load("./models/dialogue", interpreter="./models/current/nlu/default/model_20180911-134739")
# response = agent.handle_message("我正在找吃饭的地方")
# print(response)
# response = agent.handle_message("餐馆")
# print(response)
# response = agent.handle_message("万达")
# print(response)
# response = agent.handle_message("日本料理")
# print(response)


import IPython
from IPython.display import clear_output,display
from rasa_core.agent import Agent
import time

messages = ["Hi! you can chat in this window. Type 'stop' to end the conversation."]
agent = Agent.load("./models/dialogue",  interpreter="./models/current/nlu/default/model_20180911-134739")
def chatlogs_html(messages):
    messages_html = "".join(["<p>{}</p>".format(m) for m in messages])
    chatbot_html = """<div class="chat-window" {}</div>""".format(messages_html)
    return chatbot_html


while True:
    clear_output()
    display(IPython.display.HTML(chatlogs_html(messages)))
    print(messages)
    time.sleep(0.3)
    a = input()
    messages.append(a)
    if a == 'stop':
        break
    responses = agent.handle_message(a)
    for r in responses:
        messages.append(r.get("text"))