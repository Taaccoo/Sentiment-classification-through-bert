import torch 
import torch.nn as nn 
import sys
sys.path.append("..")


from transformers import BertModel,BertTokenizer


class Config(object):

    def __init__(self,dataset):
        self.model_name = "MyBert"
        self.train_path = dataset+"/data/train.txt"
        self.test_path = dataset+"/data/test.txt"
        self.dev_path = dataset+"/data/dev.txt"
        self.datasetpkl = dataset+"/data/dataset/"
        self.class_list = [x.strip() for x in open(dataset+"/data/class.txt").readlines()]

        self.save_path = dataset + "/saved_dict/"+self.model_name+'.ckpt'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        ## stop training if performance not improve less than 1000batch
        self.require_improvement = 1000

        self.num_classes = len(self.class_list)

        self.num_epochs = 3

        self.batch_size = 128

        self.pad_size = 32 

        self.learning_rate = 1e-5

        self.bert_path = "bert_pretrain"

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(config.hidden_size,config.num_classes)

        
    def forward(self,x):
        # x [ids,seq_len, mask]
        context = x[0]
        mask = x[2] #(128,32)

        # (128,768)
        output = self.bert(context,attention_mask=mask,output_hidden_states=True)
                    
        out = self.fc(output[1]) #(128,10) class_num = 10
        return out 
