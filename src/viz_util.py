import json
import os
from collections import defaultdict
import pandas as pd
import re

raw_reviews = pd.read_excel('/Users/maoc/Dropbox (MIT)/JMP/UX168 data and code/data/raw/MRO-0730/硅胶管_V2.xlsx')
review_id_to_reviews = {row['review_id']: f'<b>{row["review_title"]}</b> {row["评论内容"]}' for index, row in raw_reviews.iterrows()}
def extract_content_in_parentheses(s):
    return re.search(r'\((.*?)\)', s).group(1)
# iterate through all levels of the dict and replace the values with the ids
def get_id_to_keys(in_dict, id_to_keys=None, prev_keys=[]):
    if id_to_keys is None:
        id_to_keys = {}
    for k, v in in_dict.items():
        english_key = extract_content_in_parentheses(k)
        if isinstance(v, dict):
            get_id_to_keys(v, id_to_keys, prev_keys+[english_key])
        else:
            for id in v:
                id_to_keys[id] = prev_keys+[english_key]
    return id_to_keys

def get_path_to_ids(id_dict, last=False):
    id_to_keys = get_id_to_keys(id_dict)
    path_to_ids = defaultdict(list)
    for id, keys in id_to_keys.items():
        path = '|'.join(keys)
        if last:
            path = path.split('|')[-1]
        path_to_ids[path].append(id)
    return path_to_ids

def merge_values(val1, val2):
    if val1 is None or len(val1) == 0:
        return val2
    if val2 is None or len(val2) == 0:
        return val1
    if isinstance(val1, list) or isinstance(val2, list):
        if isinstance(val1, list) and isinstance(val2, list):
            return list(set(val1+val2))
        elif hasattr(val1, 'keys'):
            return merge_values(val1, {'_': val2})
        else:
            return merge_values({'_': val1}, val2)
    elif hasattr(val1, 'keys') and hasattr(val2, 'keys'):
        new_dict = {}
        for key in set(list(val1.keys())+list(val2.keys())):
            new_dict[key] = merge_values(val1.get(key, None), val2.get(key, None))
        return new_dict
    else:
        raise ValueError(f"Invalid value type: {type(val1)} of {val1} and {type(val2)} of {val2}")


def get_path_to_sents(path_to_ids, id_to_sent_dict):
    path_to_sents = defaultdict(dict)
    for path, ids in path_to_ids.items():
        for id in ids:
            if id not in id_to_sent_dict:
                continue
            current_sent_dict = id_to_sent_dict[id]
            sent_to_review_dict = defaultdict(dict)
            for sent, reason_rids in current_sent_dict.items():
                current_sent_dict = {}
                for reason, review_ids in reason_rids.items():
                    review_texts = [review_id_to_reviews[review_id] for review_id in review_ids]
                    sent_to_review_dict[sent][reason] = review_texts
            path_to_sents[path] = merge_values(path_to_sents[path], sent_to_review_dict)
    return path_to_sents



# Main execution
if __name__ == '__main__':
    RESULT_DIR = "/Users/maoc/Dropbox (MIT)/JMP_VIZ/result/MRO_silicon_tube"
    with open(os.path.join(RESULT_DIR, "cleaned_use_dict_it4.json"), 'r') as file:
        use_id_dict = json.load(file)

    with open(os.path.join(RESULT_DIR, "cleaned_attr_perf_dict_it4.json"), 'r') as file:
        attr_perf_dict = json.load(file)
        attr_id_dict = attr_perf_dict['attr']
        perf_id_dict = attr_perf_dict['perf']

    with open(os.path.join(RESULT_DIR, "use_id_to_values.json"), 'r') as file:
        use_id_to_sents_dict = json.load(file)

    with open(os.path.join(RESULT_DIR, "attr_id_to_values.json"), 'r') as file:
        attr_id_to_sents_dict = json.load(file)

    with open(os.path.join(RESULT_DIR, "perf_id_to_values.json"), 'r') as file:
        perf_id_to_sents_dict = json.load(file)

    use_path_to_ids_dict = get_path_to_ids(use_id_dict)
    use_path_to_sents_dict = get_path_to_sents(use_path_to_ids_dict, use_id_to_sents_dict)
    attr_path_to_ids_dict = get_path_to_ids(attr_id_dict)
    perf_path_to_ids_dict = get_path_to_ids(perf_id_dict)
    attr_perf_path_to_ids_dict = merge_values(attr_path_to_ids_dict, perf_path_to_ids_dict)
    perf_path_to_sents_dict = get_path_to_sents(perf_path_to_ids_dict, perf_id_to_sents_dict)

    json.dump(use_path_to_sents_dict, open(os.path.join(RESULT_DIR, "use_path_to_sents_dict.json"), 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    json.dump(attr_perf_path_to_ids_dict, open(os.path.join(RESULT_DIR, "attr_perf_path_to_ids_dict.json"), 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    json.dump(perf_path_to_sents_dict, open(os.path.join(RESULT_DIR, "perf_path_to_sents_dict.json"), 'w', encoding='utf8'), ensure_ascii=False, indent=2)
    json.dump(attr_path_to_ids_dict, open(os.path.join(RESULT_DIR, "attr_path_to_ids_dict.json"), 'w', encoding='utf8'), ensure_ascii=False, indent=2)