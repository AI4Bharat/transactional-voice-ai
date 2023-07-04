import pickle
import yaml

def check_intent(config):
    defined_intents = set(config["intent"]["supported_intents"])
    with open(config["intent"]["label_dict_path"], "rb") as f:
        supported_intents = set(pickle.load(f).keys())
    if defined_intents != supported_intents:
        return False
    else:
        return True


def check_entity(config):
    defined_entities = set(config["entities"]["supported_entities"])
    for lang in config["supported_languages"]:
        with open(config["entities"]["variation_path"][lang]) as f:
            variations_dict = yaml.load(f, yaml.BaseLoader)
        with open(config["entities"]["pattern_path"]) as f:
            pattern_dict = yaml.load(f, yaml.BaseLoader)
        supported_entities = set(pattern_dict.keys()) | set(variations_dict.keys())
        if defined_entities != supported_entities:
            return False
    return True
