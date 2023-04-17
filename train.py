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

from mixtext import MixText

from data import LabeledDataset, UnlabeledDataset

import gc
gc.collect()
torch.cuda.empty_cache()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_acc = 0
total_steps = 0
flag = 0

NUM_LABELS = 10 # Number of labels in Yahoo
PAD_token = 1 # RoBERTa 

def main(config_name):

    global config

    with open(os.path.join("configs", f"{config_name}.json"), "r") as f:
        config = json.load(f)

    global best_acc

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

    test_accs = []

    # Start training
    for epoch in range(config['epochs']):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
              scheduler, train_criterion, epoch, NUM_LABELS)

        # scheduler.step()

        # _, train_acc = validate(labeled_trainloader,
        #                        model,  criterion, epoch, mode='Train Stats')
        #print("epoch {}, train acc {}".format(epoch, train_acc))

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        # if val_acc >= best_acc:
        #     best_acc = val_acc
        #     test_loss, test_acc = validate(
        #         test_loader, model, criterion, epoch, mode='Test Stats ')
        #     test_accs.append(test_acc)
        #     print("epoch {}, test acc {},test loss {}".format(
        #         epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


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

    for batch_idx in range(len(unlabeled_trainloader)):

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
            p = (0 * torch.softmax(outputs_u, dim=1) + 0 * torch.softmax(outputs_u2,
                                                                         dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / (1)
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

        all_inputs = torch.cat(
            [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

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

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/len(unlabeled_trainloader), mixed)

        loss = Lx + w * Lu

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def validate(valloader, model, criterion, epoch, mode):
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
                print("Sample some true labeles and predicted labels")
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

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - config['hinge_margin'], min=0))

        elif config['mix_method'] == 2:
            if mixed == 0:
                Lx = - \
                    torch.mean(torch.sum(F.logsigmoid(
                        outputs_x) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)

                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(config['hinge_margin'] - torch.sum(
                    F.softmax(outputs_u_2, dim=1) * F.softmax(outputs_u_2, dim=1), dim=1), min=0))
            else:
                Lx = - \
                    torch.mean(torch.sum(F.log_softmax(
                        outputs_x, dim=1) * targets_x, dim=1))

                probs_u = torch.softmax(outputs_u, dim=1)
                Lu = F.kl_div(probs_u.log(), targets_u,
                              None, None, 'batchmean')

                Lu2 = torch.mean(torch.clamp(config['hinge_margin'] - torch.sum(
                    F.softmax(outputs_u, dim=1) * F.softmax(outputs_u, dim=1), dim=1), min=0))

        return Lx, Lu, config['lambda_u'] * linear_rampup(epoch, config['epochs']), Lu2, config['lambda_u_hinge'] * linear_rampup(epoch, config['epochs'])


if __name__ == '__main__':
    
    main(sys.argv[1])