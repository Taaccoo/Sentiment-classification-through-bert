import time
import torch
import numpy as np 
import utils
from torch.utils.data import DataLoader,Subset
#动态导入
from importlib import import_module 
import argparse
import train 
parser = argparse.ArgumentParser(description="MyBert-Text-Classification")
parser.add_argument('--model',type=str,default='MyBert',help="choose a model")
args = parser.parse_args()

if __name__ == "__main__":
    dataset = 'THUCNews'
    model_name = args.model 
    print(model_name)
    x = import_module('model.'+model_name)
    config = x.Config(dataset)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic=True

    start_time = time.time()
    print("loading dataset..")
    train_data = utils.THUCNews(config,0)
    dev_data = utils.THUCNews(config,1)
    test_data = utils.THUCNews(config,2)
    
    train_data_iter = DataLoader(train_data,batch_size=config.batch_size,shuffle=True)
    dev_data_iter = DataLoader(dev_data,batch_size=config.batch_size,shuffle=False)
    test_data_iter = DataLoader(test_data,batch_size=config.batch_size,shuffle=False)
    
    
    # #print(train_data.__len__())
    # train_iter = utils.build_iterator(train_data,config)
    # dev_iter = utils.build_iterator(dev_data,config)
    # test_iter= utils.build_iterator(test_data,config)
   
    
    time_dif = utils.get_time_dif(start_time)
    print("Before start- data processing time",time_dif)

    model = x.Model(config).to(config.device)
   

    train.train(config,model,train_data_iter,dev_data_iter,test_data_iter)
