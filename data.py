import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

EOS_token = '2' # RoBERTa

class LabeledDataset(Dataset):

    def __init__(self, idx_name = None):

        if idx_name is None:
            self.data = pd.read_csv(os.path.join("reduced_data", "yahoo_test.csv"), 
                                    index_col=0)
        
        else:
            idx = torch.load(os.path.join("indices", idx_name))
            self.data = pd.read_csv(os.path.join("reduced_data", "yahoo_train.csv"), 
                                    index_col=0).loc[idx]
        
    def __len__(self):

        return self.data.shape[0]
    
    def __getitem__(self, index):

        input_ids, _, label = self.data.iloc[index][['input_ids', 'attention_mask', 'labels']]

        input_ids = torch.tensor(np.array(input_ids.split()[1:-1] + [EOS_token], dtype=np.int64)) # convert string to array

        return input_ids, label, len(input_ids)


class UnlabeledDataset(Dataset):

    def __init__(self, idx_name):

        idx = torch.load(os.path.join("indices", idx_name))

        self.in_domain = pd.read_csv(os.path.join("reduced_data", "yahoo_train.csv"), 
                                     index_col=0).loc[idx['in_domain_idx']]
        self.in_domain_russian = pickle.load(open(os.path.join("all_backtranslations", 
                                                               "yahoo_train_russian.pkl"), 'rb'))
        self.in_domain_german = pickle.load(open(os.path.join("all_backtranslations", 
                                                              "yahoo_train_german.pkl"), 'rb'))
        
        self.out_of_domain = pd.read_csv(os.path.join("reduced_data", "OOD_concatenated.csv"), 
                                     index_col=0).loc[idx['out_of_domain_idx']]
        self.out_of_domain_russian = pickle.load(open(os.path.join("all_backtranslations", 
                                                                   "OOD_concatenated_russian.pkl"), 'rb'))
        self.out_of_domain_german = pickle.load(open(os.path.join("all_backtranslations", 
                                                                  "OOD_concatenated_german.pkl"), 'rb'))
        
    def __len__(self):

        return self.in_domain.shape[0] + self.out_of_domain.shape[0]
    
    def __getitem__(self, index):

        # index determines if in_domain or out_of_domain is chosen
        if index < self.in_domain.shape[0]:
            data = self.in_domain
            backtranslation_choices = [self.in_domain_russian, self.in_domain_german]
        else:
            index -= self.in_domain.shape[0] # readjust index for OOD
            data = self.out_of_domain
            backtranslation_choices = [self.out_of_domain_russian, self.out_of_domain_german]

        input_ids, _, _ = data.iloc[index][['input_ids', 'attention_mask', 'labels']]

        input_ids = torch.tensor(np.array(input_ids.split()[1:-1] + [EOS_token], dtype=np.int64)) # convert string to array

        idx = data.iloc[index].name
        ru_backtranslation, de_backtranslation = backtranslation_choices[0][idx], backtranslation_choices[1][idx]
        ru_input_ids, _ = (torch.tensor(np.array(ru_backtranslation['input_ids'])), 
                                           torch.tensor(np.array(ru_backtranslation['attention_mask'])))
        de_input_ids, _ = (torch.tensor(np.array(de_backtranslation['input_ids'])), 
                                           torch.tensor(np.array(de_backtranslation['attention_mask'])))
        
        return ru_input_ids, de_input_ids, input_ids, len(ru_input_ids), len(de_input_ids), len(input_ids)
