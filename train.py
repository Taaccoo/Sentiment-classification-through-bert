import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn import metrics 
import time 
import utils 
from transformers import AdamW,get_linear_schedule_with_warmup


def train(config,model,train_iter,dev_iter,test_iter):
    """
    train processing
    """
    start_time = time.time()
    


    para_optimizer = list(model.named_parameters())
    # 设置哪些参数不需要衰减
    no_decay = ['bais','LayerNorm.bias','LayerNorm.weight']

    optimizer_grouped_parameters=[
        {'params':[p for n,p in para_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.0},
        {'params':[p for n,p in para_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]

    #BertAdam 自定义封装了优化函数了warmup函数，也可以自己分开
    # optimizer = BertAdam(params=optimizer_grouped_parameters,
    #                         lr=config.learning_rate,
    #                         warmup=0.05,
    #                         t_total=len(train_iter)*config.num_epochs
    #                     )
    
    ## version 4x 版本将其分开，要使用AdamW和 get_linear_schedule_with_warmup
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr = 1e-5,
        eps=1e-8
    ) 

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.05,
        num_training_steps = len(train_iter)*config.num_epochs
    ) 


    total_batch= 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = False 

    model.train()
    #for epoch in range(config.epochs):
    for epoch in range(1):
        print("epoch [{}/{}]".format(epoch+1,config.num_epochs))

        for i,(train,labels) in enumerate(train_iter):
            outputs = model(train)
            model.zero_grad()
            loss = F.cross_entropy(outputs,labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()
            scheduler.step() 

            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predit = torch.max(outputs.data,1)[1].cpu()
                train_acc = metrics.accuracy_score(true,predit)
                dev_acc,dev_loss = evaluate(config,model,dev_iter,test=False)

                if dev_loss<dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(),config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_idf = utils.get_time_dif(start_time)
                msg = "Iter:{0:6}, Train Loss:{1:5.2} ,Train Acc:{2:>6.2%},Val Loss:{3:>5.2},Val Acc:{4:>6.2%},Time:{5} {6}"
                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc,time_idf,improve))

                model.train()

            total_batch = total_batch+1

            if total_batch - last_improve > config.require_improvement:
                print("long time no improve")
                flag = True 
                break
        if flag:
            break 
    
    test(config,model,test_iter)


def evaluate(config,model,dev_iter,test=False):
    model.eval()

    loss_total = 0
    predict_all = np.array([],dtype=int)
    labels_all = np.array([],dtype=int)

    with torch.no_grad():
        for text,labels in dev_iter:
            outputs = model(text)
            loss = F.cross_entropy(outputs,labels)
            loss_total = loss_total+loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all,labels)
            predict_all = np.append(predict_all,predict)

    acc = metrics.accuracy_score(labels_all,predict_all)
    if test:
        report = metrics.classification_report(labels_all,predict_all,target_names=config.class_list,digits=4)
        confusion = metrics.confusion_matrix(labels_all,predict_all)
        return acc,loss_total/len(dev_iter),report,confusion 
    return acc,loss_total/len(dev_iter)

def test(config,model,test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()

    start_time = time.time()
    test_acc,test_loss,test_report,test_confusion = evaluate(config,model,test_iter,test=True)

    msg = "Test Loss:{0:>5.2},Test Acc:{1:>6.2%}"
    print(msg.format(test_loss,test_acc))
    print("Precision,Recoall and F1-score")
    print(test_report)
    print("confusion Matrix")
    print(test_confusion)

    time_dif = utils.get_time_dif(start_time)
    print("time",time_dif)