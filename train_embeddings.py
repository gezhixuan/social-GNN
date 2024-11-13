import torch
import random
import argparse
import pickle as pickle
import torch.nn as nn
import numpy as np
import yaml

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
from embeddings import Embeddings

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Replace placeholders for DATA_HOME and LOG_DIR in paths
    data_home = config["settings"]["DATA_HOME"]
    log_dir = config["settings"]["LOG_DIR"]
    for key, value in config["paths"].items():
        config["paths"][key] = value.replace("{DATA_HOME}", data_home).replace("{LOG_DIR}", log_dir)

    return config


def load_data(batch_size, max_len, config):
    print("Loading train/test data...")
    thread_to_sub = {}
    print(config["paths"]["POST_INFO"])
    
    with open(config["paths"]["POST_INFO"]) as fp:
        for line in fp:
            info = line.split()
            source_sub = info[0]
            target_sub = info[1]
            source_post = info[2].split("T")[0].strip()
            target_post = info[6].split("T")[0].strip()
            thread_to_sub[source_post] = source_sub
            thread_to_sub[target_post] = target_sub

    label_map = {}
    source_to_dest_sub = {}
    with open(config["paths"]["LABEL_INFO"]) as fp:
        for line in fp:
            info = line.split("\t")
            source = info[0].split(",")[0].split("\'")[1]
            dest = info[0].split(",")[1].split("\'")[1]
            label_map[source] = 1 if info[1].strip() == "burst" else 0
            try:
                source_to_dest_sub[source] = thread_to_sub[dest]
            except KeyError:
                continue

    with open(config["paths"]["SUBREDDIT_IDS"]) as fp:
        sub_id_map = {sub:i for i, sub in enumerate(fp.readline().split())}

    with open(config["paths"]["USER_IDS"]) as fp:
        user_id_map = {user:i for i, user in enumerate(fp.readline().split())}
        
    # loading handcrafted features
    meta_features = {}
    meta_labels = {}
    with open(config["paths"]["META_FEATURE"]) as fp:
        for line in fp:
            info = line.split()
            meta_features[info[0]] = np.array(info[-1].split(",")).astype(np.float64)
            # print(meta_features[info[0]].shape)
            # exit(0)
            meta_labels[info[0]] = 1 if info[1] == "burst" else 0

    with open(config["paths"]["PREPROCESSED_DATA"]) as fp:
        words, users, subreddits, lengths, labels, ids, meta_features_list = [], [], [], [], [], [], []
        for i, line in enumerate(fp):
            info = line.split("\t")
            if not (info[1] in meta_features.keys()):
                continue
            if info[1] in label_map and info[1] in source_to_dest_sub:
                title_words = info[-2].split(":")[1].strip().split(",")
                title_words = title_words[:min(len(title_words), config["constants"]["MAX_LEN"])]
                if len(title_words) == 0 or title_words[0] == '':
                    continue
                words.append(list(map(int, title_words)))

                body_words = info[-1].split(":")[1].strip().split(",")
                body_words = body_words[:min(len(body_words), config["constants"]["MAX_LEN"]-len(title_words))]
                if not (len(body_words) == 0 or body_words[0] == ''):
                    words[-1].extend(list(map(int, body_words)))

                words[-1] = [config["constants"]["VOCAB_SIZE"]+1 if w==-1 else w for w in words[-1]]

                if not info[0] in sub_id_map:
                    source_sub = config["constants"]["NUM_SUBREDDITS"]
                else:
                    source_sub = sub_id_map[info[0]]
                dest_sub = source_to_dest_sub[info[1]]
                if not dest_sub in sub_id_map:
                    dest_sub = config["constants"]["NUM_SUBREDDITS"]
                else:
                    dest_sub = sub_id_map[dest_sub]
                    
                meta_features_list.append(meta_features[info[1]])

                subreddits.append([source_sub, dest_sub])

                users.append([config["constants"]["NUM_USERS"] if not info[3] in user_id_map else user_id_map[info[3]]])
                ids.append(info[1])


                lengths.append(len(words[-1])+3)
                labels.append(label_map[info[1]])

        batches = []
        np.random.seed(0)
        for count, i in enumerate(np.random.permutation(len(words))):
            if count % batch_size == 0:
                batch_words = np.ones((max_len, batch_size), dtype=np.int64) * config["constants"]["VOCAB_SIZE"]
                batch_users = np.ones((1, batch_size), dtype=np.int64) * config["constants"]["VOCAB_SIZE"]
                batch_subs = np.ones((2, batch_size), dtype=np.int64) * config["constants"]["VOCAB_SIZE"]
                batch_lengths = []
                batch_labels = []
                batch_ids = []
                batch_meta_features = np.ones((263, batch_size), dtype=np.float32) * config["constants"]["VOCAB_SIZE"]
                
            length = min(max_len, len(words[i]))
            batch_words[:length, count % batch_size] = words[i][:length]
            batch_users[:, count % batch_size] = users[i]
            batch_subs[:, count % batch_size] = subreddits[i]
            batch_meta_features[:, count % batch_size] = meta_features_list[i]
            batch_lengths.append(length)
            batch_labels.append(labels[i])
            batch_ids.append(ids[i])
            # batch_meta_features.append(meta_features_list[i])
            if count % batch_size == batch_size - 1:
                order = np.flip(np.argsort(batch_lengths), axis=0)
                batches.append((list(np.array(batch_ids)[order]),
                    torch.LongTensor(batch_words[:,order]), 
                    torch.LongTensor(batch_users[:,order]), 
                    torch.LongTensor(batch_subs[:,order]), 
                    list(np.array(batch_lengths)[order]),
                    torch.LongTensor(batch_meta_features[:,order]),
                    torch.FloatTensor(np.array(batch_labels)[order])))
    print(len(batches))
    return batches

def split_and_save_data(data, config):
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    num_data = len(data)
    
    train_end = int(train_ratio * num_data)
    val_end = train_end + int(val_ratio * num_data)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    with open(config["paths"]["TRAIN_DATA"], 'wb') as f:
        pickle.dump(train_data, f)
    with open(config["paths"]["VAL_DATA"], 'wb') as f:
        pickle.dump(val_data, f)
    with open(config["paths"]["TEST_DATA"], 'wb') as f:
        pickle.dump(test_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data and configuration.")
    parser.add_argument("--config", type=str, default="configs/train_embeddings.yaml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load the data as a single list
    data = load_data(512, config["constants"]["MAX_LEN"], config)

    # Split the data and save to separate files
    split_and_save_data(data, config)

