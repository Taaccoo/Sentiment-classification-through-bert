import torch 
import torch.nn as nn
import torch.utils.data.dataset as Data  
import pickle as pkl 
from tqdm import tqdm 
import time
from datetime import timedelta
import os 



class THUCNews(Data.Dataset):
    def __init__(self,config,select_set):
        """ 
            select_set：
            0 : train
            1 : dev 
            2 : test 

        """
        ## [token_ids,int(label),seq_len,mask]
        self.dataset = self.get_dataset(config,select_set)

        self.x = torch.LongTensor([item[0] for item in self.dataset]).to(config.device) 
        self.y = torch.LongTensor([item[1] for item in self.dataset]).to(config.device) 
        self.seq_len = torch.LongTensor([item[3] for item in self.dataset]).to(config.device) 
        self.mask = torch.LongTensor([item[3] for item in self.dataset]).to(config.device)

        

    def __getitem__(self,ids):
        return (self.x[ids],self.seq_len[ids],self.mask[ids]),self.y[ids]
 
    def __len__(self):
        return self.dataset.__len__()

    def get_dataset(self,config,select_set):
        """
            select_set：
            0 : train
            1 : dev 
            2 : test 

            returns:
                dataset dict 
        """
        datatype = "train"
        path = config.train_path

        if select_set == 1:
            datatype = "dev"
            path = config.dev_path
        elif select_set == 2:
            datatype = "test"
            path = config.test_path 
        
        if os.path.exists(config.datasetpkl+datatype+".pkl"):
            dataset = pkl.load(open(config.datasetpkl+datatype+".pkl",'rb'))
            
        else: 
            dataset= self.load_dataset(path,config)
            pkl.dump(dataset,open(config.datasetpkl+datatype+".pkl",'wb'))
        return dataset 

    def load_dataset(self,file_path,config):
        """
        returns:
            4 list  ids,label,ids_len,mask
        """
        print(config.tokenizer)
        contents = []
        with open(file_path,'r',encoding='UTF-8') as f: 
            for line in tqdm(f):
                line = line.strip()
                if not line:
                    continue
                content,label = line.split("\t")
                token = config.tokenizer.tokenize(content)
                token = ["[CLS]"] + token

                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                
                pad_size = config.pad_size

                if pad_size:
                    if len(token)<pad_size:
                        mask = [1]*len(token_ids)+[0]*(pad_size-len(token))
                        token_ids = token_ids+([0]*(pad_size-len(token)))
                    else:
                        mask = [1]*pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append([token_ids,int(label),seq_len,mask])

        return contents 



def get_time_dif(start_time):
    """
    get running time
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == "__main__()":
    DatasetIterator(1,2,3)