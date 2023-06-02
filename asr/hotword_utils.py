import yaml

BASE_HOTWORD_PATH = "asr/data/hotwords/{}.txt"
ENTITY_VARIATIONS_PATH = "entity/data/variations/variations-{}.yaml"


def hotword_process(word):
    word = word.strip().lower()
    return word


def get_entity_variations(lang):
    with open(ENTITY_VARIATIONS_PATH.format(lang)) as f:
        entities = yaml.load(f, yaml.BaseLoader)
    variations = list()
    for ent_type, variation_dict in entities.items():
        for ent_value, variation_list in variation_dict.items():
            variations.extend(variation_list)
    return variations


def get_base_hotwords(lang):
    with open(BASE_HOTWORD_PATH.format(lang), "r") as f:
        hotwords = f.read().splitlines()
    hotwords = sorted(list(map(hotword_process, hotwords)))
    return hotwords


def get_entity_unique_hotwords(lang):
    base_hw = get_base_hotwords(lang)
    entity_variations = get_entity_variations(lang)
    unique_words = list(set(" ".join(entity_variations).split()))
    unique_words = sorted([w for w in unique_words if len(w) > 3])
    hotwords = base_hw + unique_words
    return hotwords


def get_entity_whole_hotwords(lang):
    base_hw = get_base_hotwords(lang)
    entity_variations = get_entity_variations(lang)
    entity_variations = sorted(list(set([w for w in entity_variations])))
    hotwords = base_hw + entity_variations
    return hotwords


hotword_to_fn = {
    "base": get_base_hotwords,
    "entities-unique": get_entity_unique_hotwords,
    "entities-whole": get_entity_whole_hotwords,
}
