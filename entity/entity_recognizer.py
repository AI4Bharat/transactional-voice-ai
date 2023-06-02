import re

import yaml

ENTITY_VARIATION_PATH = "entity/data/variations/variations-{}.yaml"
ENTITY_PATTERN_PATH = "entity/data/patterns.yaml"


def intersection_check(start1, end1, start2, end2):
    if start1 <= start2 <= end1 or start1 <= end2 <= end1:
        return True
    else:
        return False


class EntityRecognizer:
    def __init__(self, lang):
        self.lang = lang
        with open(ENTITY_VARIATION_PATH.format(lang)) as f:
            self.variations_dict = yaml.load(f, yaml.BaseLoader)
        with open(ENTITY_PATTERN_PATH) as f:
            self.pattern_dict = yaml.load(f, yaml.BaseLoader)
        self.patterns = dict()
        for entity_type, pattern_list in self.pattern_dict.items():
            if len(pattern_list) == 1:
                pattern = pattern_list[0]
            else:
                pattern = "|".join(pattern_list)
            self.patterns[entity_type] = pattern

    def create_entity_dict_from_match(self, ent_type, ent_word, ent_val, start, end):
        return {
            "entity": ent_type,
            "word": ent_word,
            "value": ent_val,
            "start": start,
            "end": end,
        }

    def predict(self, sentence, sentence_itn=None):
        if not sentence_itn:
            sentence_itn = sentence
        entities = list()
        for ent_type in self.variations_dict:
            for ent_val, ent_variations in self.variations_dict[ent_type].items():
                for variation in ent_variations:
                    variation_pattern = r"\b{}\b".format("\s*".join(list(variation)))
                    match = list(re.finditer(variation_pattern, sentence))
                    if not match:
                        continue
                    for m in match:
                        entities.append(
                            self.create_entity_dict_from_match(
                                ent_type, variation, ent_val, m.start(), m.end()
                            )
                        )

        for ent_type, pattern in self.patterns.items():
            match = list(re.finditer(pattern, sentence_itn))
            for m in match:
                entities.append(
                    self.create_entity_dict_from_match(
                        ent_type, m.group(0), m.group(0), m.start(), m.end()
                    )
                )

        filtered_entities = self.remove_overlap(entities)
        final_entities = self.format_entities(filtered_entities)
        return final_entities

    def remove_overlap(self, entities):
        entities = sorted(entities, key=lambda x: x["end"] - x["start"], reverse=True)
        filtered_entities = list()
        for i, ent in enumerate(entities):
            intersects = False
            for j, f_ent in enumerate(filtered_entities):
                if intersection_check(
                    ent["start"], ent["end"], f_ent["start"], f_ent["end"]
                ):
                    intersects = True
            if not intersects:
                filtered_entities.append(ent)

        return filtered_entities

    def format_amount(self, ent_val):
        remove_rupees_substring = [
            "rupees",
            "rs",
            "रूपीस",
            "रुपीस",
            "रूपी",
            "रुपी",
            "रूपये",
            "रुपये",
            "रूपय",
            "रुपय",
            "रूपए",
            "रुपए",
            "₹",
        ]
        for ss in remove_rupees_substring:
            ent_val = ent_val.replace(ss, "")
        ent_val = ent_val.strip()
        return ent_val

    def format_entities(self, entities):
        fn_dict = {
            "amount_of_money": self.format_amount,
            "mobile_number": lambda x: "".join(x.strip().split()),
            "vehicle_number": lambda x: "".join(x.strip().split()),
        }

        entities_formatted = list()
        for ent in entities:
            if ent["entity"] in fn_dict:
                ent["value"] = fn_dict[ent["entity"]](ent["value"])
                entities_formatted.append(ent)
            else:
                entities_formatted.append(ent)
        return entities_formatted
