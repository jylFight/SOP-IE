import os
import torch
import json
import argparse
from transformers import AutoConfig, AutoModel
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import dealdata
import models
import ON_LSTM
import utils

def evaluate(model,feature_data,device,tags:str,model_type:str,pos=False):    #考虑分词的
    assert model_type in ['enti','rel']
    preds = []
    rel_true = []
    if model_type=='enti':
        dataloader = DataLoader(feature_data, batch_size=8, shuffle=False, collate_fn=utils.enti_collate_fn, drop_last=True)
    if model_type=='rel':
        dataloader = DataLoader(feature_data, batch_size=4, shuffle=False, collate_fn=utils.rel_collate_fn, drop_last=True)
    for batch in dataloader:
        model.eval()
        if model_type=='enti':
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'enti_mask':batch[4].to(device),
                      'enti_size':batch[3].to(device), 'crf_mask': batch[5].to(device),}
        if pos:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],'pos_info':batch[8].to(device),}
        else:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],}
            
            rel_true.extend(batch[7])
        with torch.no_grad():
            output = model(**inputs)     
            preds.extend(output[0])
    
    #这里注意一下，要把0项排除才行，这里需要计算一下，可能需要手动计算,这里就直接算micro-f1了
    fenzi,p_fenmu,r_fenmu=0,0,0
    if model_type=='enti': 
        for num in range(len(preds)):        
            y_true=feature_data[num]['enti_tags']
            y_pred=preds[num].argmax(dim=-1)
            for i in range(len(y_true)):
                if y_true[i]==y_pred[i] and y_true[i]>0:
                    fenzi += 1
            p_fenmu += ((np.array(y_true))>0).sum()
            new_y_pred = y_pred[:len(y_true)]          #把padding的排除
            r_fenmu += ((np.array(new_y_pred.cpu()))>0).sum()
            
    if model_type=='rel':
        for num in range(len(preds)):                  #这里换一下，只考虑关系词的判定来看看结果
            y_true=(rel_true[num]>4).float()
            y_pred=(torch.tensor(preds[num])>4).float()
            for i in range(len(y_true)):
                for j in range(len(y_true[i])):
                    if y_true[i][j]==y_pred[i][j] and y_true[i][j]>0:
                        fenzi+=1
            p_fenmu += ((np.array(y_true))>0).sum()
            r_fenmu += ((np.array(y_pred))>0).sum()       
    
    p=fenzi/p_fenmu
    if r_fenmu==0:
        r=0
    else:
        r=fenzi/r_fenmu
    result = 2*p*r/(p+r)
    print(model_type,"_model的P值为:",p,"R值为:",r,"fenzi为:",fenzi,"p_fenzi为:",p_fenmu,"r_fenmu为:",r_fenmu)
    print(tags+'_F1:',result)
    return result


def train(model, train_features, dev_features,device,train_batch_size:int,num_epoch, learning_rate,model_type,model_name,pos=False,neg=False):
    def finetune(features, optimizer, num_epoch,model_type,model_name,pos):
        best_score = -1
        if model_type=='enti':
            train_dataloader = DataLoader(features, batch_size=train_batch_size, shuffle=True, collate_fn=utils.enti_collate_fn, drop_last=True)
        if model_type=='rel':
            train_dataloader = DataLoader(features, batch_size=train_batch_size, shuffle=True, collate_fn=utils.rel_collate_fn, drop_last=True)
        total_steps = int(len(train_dataloader) * num_epoch )
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        num_steps=0
        for epoch in range(int(num_epoch)):
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                if model_type=='enti':
                    inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'enti_mask':batch[4].to(device),
                          'enti_size':batch[3].to(device),'enti_tags': batch[2].to(device),'crf_mask': batch[5].to(device),}  
                    
                if model_type=='rel':
                    if pos:
                        inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                              'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],
                            'rel_tags':batch[6].to(device),'pos_info':batch[8].to(device),}
                    else:
                        inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                                  'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],
                                'rel_tags':batch[6].to(device)} 
                outputs = model(**inputs)
                loss = outputs[0] 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                if step%50==0:
                    print("num_step:",step)
                num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 :
                    print("第",epoch,"轮loss为:",loss)
                    dev_F1= evaluate(model, dev_features,device, tags="dev",model_type=model_type)
                    if dev_F1 > best_score:
                        best_score = dev_F1
                        if neg==False:
                            torch.save(model.state_dict(),'model_result/'+model_name+'_model.pth' )
                        else:
                            torch.save(model.state_dict(),'model_result/'+model_name+'_model_neg.pth' )
        return num_steps
    
    assert model_type in ['enti','rel']
    if model_type=='enti':
        new_layer = ["FNN", "size_embeddings"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], "weight_decay":0.01 },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": learning_rate},
            ]
    if model_type=='rel':       
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if 'model' in n ], "weight_decay":0.01 },
            {"params": [p for n, p in model.named_parameters() if not 'model' in n ], "lr": learning_rate},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
    finetune(train_features, optimizer, num_epoch, model_type,model_name,pos)

def main(args):
    utils.set_seed(10)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    assert data_type in ['LSOIE','RE-OIE2016']

    with open('data/LSOIE_train.json','r') as f:
        train_data = json.load(f)
    with open('data/LSOIE_dev.json','r') as f:
        dev_data = json.load(f)
    if args.data_type=='LSOIE':
        with open('data/LSOIE_test.json','r') as f:
            test_data = json.load(f)
    else:
        with open('data/deal_data.json','r') as f:
            test_data = json.load(f)
    
    train_data=dealdata.deal_conll(train_data)
    dev_data=dealdata.deal_conll(dev_data)
    test_data=dealdata.deal_conll(test_data) if args.data_type=='LSOIE' else test_data 
    
    train_feature=dealdata.dealdata(train_data,8,40,train=True,sci=False,pos_file='data/pos/train.txt')
    dev_feature=dealdata.dealdata(dev_data,8,40,train=False,sci=False,pos_file='data/pos/dev.txt')
    if args.data_type=='LSOIE': 
        test_feature=dealdata.dealdata(test_data,8,40,train=False,sci=False,pos_file='data/pos/test.txt')
    else:
        test_feature=dealdata.dealdata(test_data,8,40,train=False,sci=False,pos_file='data/pos/RE_test.txt')
    train_neg_feature=utils.neg_data(train_feature)
    
    print('===============Start Training===============')
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")   #0有人用
    config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
    model=AutoModel.from_pretrained('microsoft/deberta-v3-base',config=config,)
    if arges.model_type=='enti':
        train_model=models.enti_bert_softmax(config,model,4)
    else:
        train_model=models.bert_bi_lstm_pure(config,model,6,False)
    train_model.to(device)
    if neg:
        train(train_model,train_neg_feature,dev_feature,device,args.batch_size,args.epochs,args.lr,args.model_type,args.model_name,neg=True)
    else:
        train(train_model,train_neg_feature,dev_feature,device,args.batch_size,args.epochs,args.lr,args.model_type,args.model_name)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)   
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--model_name', default='SOR-IE', type=str)   
    parser.add_argument('--neg', type=bool, default=True)
    parser.add_argument('--pos', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='rel')
    parser.add_argument('--data_type', type=str, default='LSOIE')
    
    args = parser.parse_args()

    main(args)
        