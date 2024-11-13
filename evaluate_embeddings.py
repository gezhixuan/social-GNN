import torch
import random
import argparse
import pickle
import torch.nn as nn
import numpy as np
import yaml

from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

from embeddings import Embeddings
from models.networks import SocialLSTM

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Replace placeholders for DATA_HOME, LOG_DIR, and EMBEDDINGS in paths
    data_home = config["settings"]["DATA_HOME"]
    log_dir = config["settings"]["LOG_DIR"]
    embeddings_dir = config["settings"]["EMBEDDINGS"]
    for key, value in config["paths"].items():
        config["paths"][key] = value.replace("{DATA_HOME}", data_home).replace("{EMBEDDINGS}", embeddings_dir).replace("{LOG_DIR}", log_dir)

    return config


def train(model, train_data, val_data, test_data, optimizer, config):
    epochs = config['training']['epochs']
    log_file = config['paths']['log_file']
    log_every = config['training']['log_every']
    save_embeds = config['training']['save_embeds']
    
    if not log_file is None:
        lg_str = log_file
        log_file = open(log_file, "w")

    criterion = nn.BCEWithLogitsLoss()
    best_iter = (0., 0, 0)
    ema_loss = None
    for epoch in range(epochs):
        random.shuffle(train_data)
        for i, batch in enumerate(train_data):
            _, text, users, subs, lengths, metafeats, labels = batch
            # print("---------------------------")
            # print(metafeats)
            text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
            optimizer.zero_grad()
            outputs = model(text, users, subs, metafeats, lengths)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss = 0.01*loss.data.item() + 0.99*ema_loss

            if i % 10 == 0:
                print(epoch, i, ema_loss)
                print(epoch, i, ema_loss, file=log_file)
            if i % log_every == 0:
                auc = evaluate_auc(model, val_data, config)
                print(f"Val AUC {epoch}, {i}: {auc}")
                if log_file:
                    print(f"Val AUC {epoch}, {i}: {auc}", file=log_file)
                if auc > best_iter[0]:
                    best_iter = (auc, epoch, i)
                    print("New best val!", best_iter)
                    best_test = evaluate_auc(model, test_data, config)
                    if auc > 0.7:
                        ids, embeds = get_embeddings(train_data+val_data+test_data)


    print("Overall best val:", best_iter)
    if not log_file is None:
        print("Overall best test:", best_test, file=log_file)
        print("Overall best val:", best_iter, file=log_file)
        if not embeds is None and save_embeds:
            np.save(open(lg_str+"-embeds.npy", "w"), embeds)
            pickle.dump(ids, open(lg_str+"-ids.pkl", "w"))
    if log_file:
        log_file.close()
    return best_iter[0]
        
def get_embeddings(data):

    embeds = []
    ids = []
    for batch in data:
        id, text, users, subs, lengths, metafeats, labels = batch
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(metafeats), Variable(labels)
        model(text, users, subs, metafeats, lengths)
        batch_embeds = model.h
        embeds.append(batch_embeds.t().data.cpu().numpy())
        ids.extend(id)
    return ids, np.concatenate(embeds)


def evaluate_auc(model, test_data, config):

    predictions = []
    gold_labels = []
    for batch in test_data:
        _, text, users, subs, lengths, metafeats, labels = batch
        if config['settings']['CUDA']:
            gold_labels.extend(labels.cpu().numpy().tolist())
        else:
            gold_labels.extend(labels.numpy().tolist())
        text, users, subs, metafeats, labels = Variable(text), Variable(users), Variable(subs), Variable(
            metafeats), Variable(labels)
        outputs = model(text, users, subs, metafeats, lengths)
        if config['settings']['CUDA']:
            predictions.extend(outputs.data.squeeze().cpu().numpy().tolist())
        else:
            predictions.extend(outputs.data.squeeze().numpy().tolist())

    auc = roc_auc_score(gold_labels, predictions)
    return auc

def move_batches_to_cuda(data):
    return [(tuple(item.cuda() if hasattr(item, 'cuda') else item for item in batch)) for batch in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/evaluate_embeddings.yaml",
                        help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    print("Loading training data")
    train_data = pickle.load(open(config['paths']['TRAIN_DATA'], 'rb'))
    val_data = pickle.load(open(config['paths']['VAL_DATA'], 'rb'))
    test_data = pickle.load(open(config['paths']['TEST_DATA'], 'rb'))
    print(f"train dataset len:{len(train_data)}")
    print(f"val dataset len:{len(val_data)}")
    print(f"test dataset len:{len(test_data)}")
    # Move data to CUDA if required
    if config["settings"]["CUDA"]:
        train_data = move_batches_to_cuda(train_data)
        val_data = move_batches_to_cuda(val_data)
        test_data = move_batches_to_cuda(test_data)

    model = SocialLSTM(config, prepend_social=not config['training']['lstm_no_social'])
    if config['settings']['CUDA']:
        model.cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['learning_rate'])
    train(model, train_data, val_data, test_data, optimizer, config)
