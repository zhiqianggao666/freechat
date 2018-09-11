# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.events import SlotSet
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

logger = logging.getLogger(__name__)

support_search = ["餐馆", "餐厅","吃饭的地方","就餐的地方","就餐","吃饭","饿了","想吃东西"]

def extract_item(item):
    if item is None:
        return None
    for name in support_search:
        if name in item:
            return name
    return None


class ActionSearchRestaurant(Action):
    def name(self):
        return 'action_search_restaurant'

    def run(self, dispatcher, tracker, domain):
        item = tracker.get_slot("item")
        item = extract_item(item)
        if item is None:
            dispatcher.utter_message("我现在只会根据地点和菜系查询餐馆信息")
            return []

        location = tracker.get_slot("location")
        #print('111111111111111111111111',location)
        if location is None:
            dispatcher.utter_message("想查询哪里的餐厅？")
            return []

        cuisines = tracker.get_slot("cuisines")
        #print('111111111111111111111111',cuisines)
        if cuisines is None:
            dispatcher.utter_message("喜欢什么菜系？")
            return []
        # query database here using item and time as key. but you may normalize time format first.
        dispatcher.utter_message("稍等")
        #dispatcher.utter_message("已经解决了%s的问题，地点%s，菜系%s".format(item, location, cuisines))
        return []


class MobilePolicy(KerasPolicy):
    def model_architecture(self, num_features, num_actions, max_history_len):
        """Build a Keras model and return a compiled model."""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        batch_shape = (None, max_history_len, num_features)

        model = Sequential()
        model.add(Masking(-1, batch_input_shape=batch_shape))
        model.add(LSTM(n_hidden, batch_input_shape=batch_shape))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation("softmax"))

        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

        logger.debug(model.summary())
        return model


def train_dialogue(domain_file="dm_domain.yml",
                   model_path="./models/dialogue",
                   training_data_file="dm_story.md"):
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy()])

    agent.train(
        training_data_file,
        epochs=200,
        batch_size=16,
        augmentation_factor=50,
        validation_split=0.2
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.training_data import load_data
    from rasa_nlu import config
    from rasa_nlu.model import Trainer

    training_data = load_data("nlu.md")
    trainer = Trainer(config.load("nlu.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist("./models/current/nlu")

    return model_directory



def run(serve_forever=True):
    agent = Agent.load("./models/dialogue",   interpreter="./models/current/nlu/default/model_20180911-134739")

    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser(
        description="starts the bot")

    parser.add_argument(
        "task",
        choices=["train-nlu", "train-dm", "run", "online_train"],
        help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task
    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dm":
        train_dialogue()
    elif task == "run":
        run()
    else:
        warnings.warn("Need to pass either 'train-nlu', 'train-dialogue' or "
                      "'run' to use the script.")
        exit(1)
