import os
import sys
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import *
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from glob import glob
import time
import pickle

from mixtext import MixText

from data import LabeledDataset, UnlabeledDataset

import gc
gc.collect()
torch.cuda.empty_cache()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

total_steps = 0
flag = 0

NUM_LABELS = 10 # Number of labels in Yahoo
PAD_token = 1 # RoBERTa 

def main(config_name):

    global config

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    # Read dataset and build dataloaders
    labeled_train = LabeledDataset(config['train_labeled_idx_name'])
    unlabeled_train = UnlabeledDataset(config['train_unlabeled_idx_name'])
    val_dataset = LabeledDataset()
    
    labeled_batch_size = get_batch_size(len(labeled_train), config)
    labeled_trainloader = DataLoader(labeled_train, 
                                     batch_size = labeled_batch_size, 
                                     shuffle = True,
                                     collate_fn = collate_batch)
    
    unlabeled_batch_size = get_batch_size(len(unlabeled_train), config)
    unlabeled_trainloader = DataLoader(unlabeled_train, 
                                        batch_size = unlabeled_batch_size, 
                                        shuffle = True,
                                        collate_fn = collate_batch)
    val_loader = DataLoader(val_dataset, 
                            batch_size = config['val_batch_size'], 
                            shuffle=False,
                            collate_fn = collate_batch)
        
    # Define the model, set the optimizer
    model = MixText(NUM_LABELS).cuda()
    model = nn.DataParallel(model)
    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": config['lrmain']},
            {"params": model.module.linear.parameters(), "lr": config['lrlast']},
        ])

    # num_warmup_steps = math.floor(50)

    scheduler = None
    #WarmupConstantSchedule(optimizer, warmup_steps=num_warmup_steps)

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    
    config_results_path = os.path.join("results", config_name)
    os.makedirs(config_results_path, exist_ok=True)

    num_results = len(glob(os.path.join(config_results_path, f"run_*")))
    run_results_path = os.path.join(config_results_path, f"run_{num_results + 1}")
    os.makedirs(run_results_path)

    # Log part before start training: -- YQ
    logger = open(os.path.join(run_results_path, 'std.log'), 'w')
        
    logger.write(f"Labeled Batch Size = {labeled_batch_size}")

    num_labeled_one_epoch = labeled_batch_size * (len(unlabeled_train) // unlabeled_batch_size) / len(labeled_train)
    logger.write(f"\nNumber of epochs through labeled data = {config['epochs'] * num_labeled_one_epoch}")

    logger.write(f"\nUnlabeled Batch Size = {unlabeled_batch_size}")
    logger.write(f"\nNumber of epochs through unlabeled data = {config['epochs']}\n")

    start_time = time.time()

    with open(os.path.join(run_results_path, "history.pkl"), 'wb') as f:
        logs = {'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []}
        pickle.dump(logs, f)
    
    # Start training
    for epoch in range(config['epochs']):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, NUM_LABELS)

        # scheduler.step()

        train_loss, train_acc = validate(labeled_trainloader, model, criterion)
        logger.write(f"\nepoch {epoch}, train acc {train_acc:.4f}, train_loss {train_loss:.4f}")

        val_loss, val_acc = validate(val_loader, model, criterion)

        logger.write(f"\nepoch {epoch}, val acc {val_acc:.4f}, val_loss {val_loss:.4f}")
        
        # Adding logs and saving models -- YQ
        logs = pickle.load(open(os.path.join(run_results_path, "history.pkl"), 'rb'))
        logs['train_loss'].append(train_loss)
        logs['train_acc'].append(train_acc)
        logs['val_loss'].append(val_loss)
        logs['val_acc'].append(val_acc)
        
        pickle.dump(logs, open(os.path.join(run_results_path, "history.pkl"), 'wb'))

        torch.save(model.state_dict(), os.path.join(run_results_path, f"epoch_{epoch + 1}.pt"))
        # End adding logs and saving models -- YQ
    
    # Log part after the end of training: -- YQ
    end_time = time.time()

    training_time_hours, remainder_secs = divmod(end_time - start_time, 3600)
    training_time_minutes, remainder_secs = divmod(remainder_secs, 60)
    
    logger.write(f"\nFinished training in: {int(training_time_hours)} hours, {int(training_time_minutes)} minutes")
    logger.close()
    # End Log part after the end of training: -- YQ


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, criterion, epoch, n_labels):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global total_steps
    global flag
    if flag == 0 and total_steps > config['temp_change']:
        print('Change T!')
        config['T'] = 0.9
        flag = 1

    for batch_idx in tqdm(range(len(unlabeled_trainloader))):
        
        if batch_idx == 100:
            break

        total_steps += 1

        try:
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, inputs_x_length = next(labeled_train_iter)
        
        try:
            (inputs_u, inputs_u2,  inputs_ori), (length_u,
                                                 length_u2,  length_ori) = next(unlabeled_train_iter) 
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_ori), (length_u,
                                                length_u2, length_ori) = next(unlabeled_train_iter)

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            outputs_ori = model(inputs_ori)

            # Based on translation qualities, choose different weights here.
            # For AG News: German: 1, Russian: 0, ori: 1
            # For DBPedia: German: 1, Russian: 1, ori: 1
            # For IMDB: German: 0, Russian: 0, ori: 1
            # For Yahoo Answers: German: 1, Russian: 0, ori: 1 / German: 0, Russian: 0, ori: 1
            p = (1 * torch.softmax(outputs_u, dim=1) + 1 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (3)
            # Do a sharpen here.
            pt = p**(1/config['T'])
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1

        l = np.random.beta(config['alpha'], config['alpha'])
        if config['separate_mix']:
            l = l
        else:
            l = max(l, 1-l)

        mix_layer = np.random.choice(config['mix_layers_set'], 1)[0]
        mix_layer = mix_layer - 1
        
        max_dim = max(inputs_x.shape[1], inputs_u.shape[1], inputs_u2.shape[1], inputs_ori.shape[1])
        all_inputs = torch.cat([torch.nn.functional.pad(inputs_x, (0, max_dim-inputs_x.shape[1], 0, 0), 'constant', 0),
                                torch.nn.functional.pad(inputs_u, (0, max_dim-inputs_u.shape[1], 0, 0), 'constant', 0),
                                torch.nn.functional.pad(inputs_u2, (0, max_dim-inputs_u2.shape[1], 0, 0), 'constant', 0),
                                torch.nn.functional.pad(inputs_ori, (0, max_dim-inputs_ori.shape[1], 0, 0), 'constant', 0),
                                torch.nn.functional.pad(inputs_ori, (0, max_dim-inputs_ori.shape[1], 0, 0), 'constant', 0)], dim=0)

        all_lengths = torch.cat(
            [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

        all_targets = torch.cat(
            [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        if config['separate_mix']:
            idx1 = torch.randperm(batch_size)
            idx2 = torch.randperm(all_inputs.size(0) - batch_size) + batch_size
            idx = torch.cat([idx1, idx2], dim=0)

        else:
            idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
            idx2 = torch.arange(batch_size_2) + \
                all_inputs.size(0) - batch_size_2
            idx = torch.cat([idx1, idx2], dim=0)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]

        if config['mix_method'] == 0:
            # Mix sentences' hidden representations
            logits = model(input_a, input_b, l, mix_layer)
            mixed_target = l * target_a + (1 - l) * target_b

        elif config['mix_method'] == 1:
            # Concat snippet of two training sentences, the snippets are selected based on l
            # For example: "I lova you so much" and "He likes NLP" could be mixed as "He likes NLP so much".
            # The corresponding labels are mixed with coefficient as well
            mixed_input = []
            if l != 1:
                for i in range(input_a.size(0)):
                    length1 = math.floor(int(length_a[i]) * l)
                    idx1 = torch.randperm(int(length_a[i]) - length1 + 1)[0]
                    length2 = math.ceil(int(length_b[i]) * (1-l))
                    if length1 + length2 > 256:
                        length2 = 256-length1 - 1
                    idx2 = torch.randperm(int(length_b[i]) - length2 + 1)[0]
                    try:
                        mixed_input.append(
                            torch.cat((input_a[i][idx1: idx1 + length1], torch.tensor([102]).cuda(), input_b[i][idx2:idx2 + length2], torch.tensor([0]*(256-1-length1-length2)).cuda()), dim=0).unsqueeze(0))
                    except:
                        print(256 - 1 - length1 - length2,
                              idx2, length2, idx1, length1)

                mixed_input = torch.cat(mixed_input, dim=0)

            else:
                mixed_input = input_a

            logits = model(mixed_input)
            mixed_target = l * target_a + (1 - l) * target_b

        elif config['mix_method'] == 2:
            # Concat two training sentences
            # The corresponding labels are averaged
            if l == 1:
                mixed_input = []
                for i in range(input_a.size(0)):
                    mixed_input.append(
                        torch.cat((input_a[i][:length_a[i]], torch.tensor([102]).cuda(), input_b[i][:length_b[i]], torch.tensor([0]*(512-1-int(length_a[i])-int(length_b[i]))).cuda()), dim=0).unsqueeze(0))

                mixed_input = torch.cat(mixed_input, dim=0)
                logits = model(mixed_input, sent_size=512)

                #mixed_target = torch.clamp(target_a + target_b, max = 1)
                mixed = 0
                mixed_target = (target_a + target_b)/2
            else:
                mixed_input = input_a
                mixed_target = target_a
                logits = model(mixed_input, sent_size=256)
                mixed = 1

        Lx, Lu, w, _, _ = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/len(unlabeled_trainloader), mixed)

        loss = Lx + w * Lu

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 20 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item()))


def validate(valloader, model, criterion):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labels and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total

def collate_batch(batch):
    """
    Labeled batch: input_ids, attention_mask, labels
    Unlabeled batch: input_ids, attention_mask, aug_input_ids, aug_attention_mask
    """
    if len(batch[0]) == 3:

        input_ids, labels, lengths = [], [], []
        for (_input, _label, _length) in batch:
            input_ids.append(_input)
            labels.append(_label)
            lengths.append(_length)
        
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = PAD_token)
                
        return input_ids, torch.tensor(labels), torch.tensor(lengths)
    
    else:

        ru_input_ids, de_input_ids, input_ids, ru_lengths, de_lengths, lengths = [], [], [], [], [], []
        for _ru_input, _de_input, _input, _ru_length, _de_length, _length in batch:
            ru_input_ids.append(_ru_input)
            de_input_ids.append(_de_input)
            input_ids.append(_input)

            ru_lengths.append(_ru_length)
            de_lengths.append(_de_length)
            lengths.append(_length)
        
        ru_input_ids = pad_sequence(ru_input_ids, batch_first = True, padding_value = PAD_token)
        de_input_ids = pad_sequence(de_input_ids, batch_first = True, padding_value = PAD_token)
        input_ids = pad_sequence(input_ids, batch_first = True, padding_value = PAD_token)
               
        return (ru_input_ids, de_input_ids, input_ids), (torch.tensor(ru_lengths), torch.tensor(de_lengths), torch.tensor(lengths))

def get_batch_size(num_samples, config):
    
    batch_size = min(num_samples//config['steps_per_epoch'], config['max_batch_size'])
    batch_size = max(batch_size, config['min_batch_size'])
    return batch_size

def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

        if config['mix_method'] == 0 or config['mix_method'] == 1:

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            log_probs_u = torch.log_softmax(outputs_u, dim=1)

            Lu = F.kl_div(log_probs_u, targets_u, None, None, 'batchmean')

            # Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
            #                                        * F.log_softmax(outputs_u, dim=1), dim=1) - config['hinge_margin'], min=0))

        elif config['mix_method'] == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                # Lu2 = torch.mean(torch.clamp(config['hinge_margin'] - torch.sum(
                #     F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                # Lu2 = torch.mean(torch.clamp(config['hinge_margin'] - torch.sum(
                #     F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, config['lambda_u'] * linear_rampup(epoch, config['epochs']), 0, 0# Lu2, config['lambda_u_hinge'] * linear_rampup(epoch, config['epochs'])


if __name__ == '__main__':
    
    main(sys.argv[1])