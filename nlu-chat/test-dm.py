from rasa_core.agent import Agent

agent = Agent.load("./models/dialogue",
                   interpreter="./models/current/nlu/default/model_20180911-134739")
response = agent.handle_message("我正在找吃饭的地方")
print(response)
response = agent.handle_message("餐馆")
print(response)
response = agent.handle_message("万达")
print(response)
response = agent.handle_message("日本料理")
print(response)