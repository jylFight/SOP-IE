import numpy as np
import torch
import copy
from torch.utils.data import DataLoader
import json
import utils
#测试用的
import ON_LSTM
import models
import dealdata
from transformers import AutoConfig, AutoModel
import argparse


#上面是之前自己的，现在是新的
def deal_span(span_data):
    temp=span_data.replace('\n','')
    span=temp.split('|')
    a1,a2,a1_score,a2_score=[],[],[],[]
    if span==['']:
        return a1,a2,a1_score,a2_score
    for data in span:
        temp_data=data.split(' ')
        span_list=temp_data[0].split(',')
        if temp_data[1]=='head,':
            a1.append(span_list)
            a1_score.append(float(temp_data[2]))
        if temp_data[1]=='tail,':
            a2.append(span_list)
            a2_score.append(float(temp_data[2]))
        if temp_data[1]=='both,':
            a1.append(span_list)
            a2.append(span_list)
            a1_score.append(float(temp_data[2]))
            a2_score.append(float(temp_data[2]))
    return a1,a2,a1_score,a2_score

def deal_enti_result(span_data,testfeature,pos=False):
    result=[]
    for i in range(len(span_data)):
        tags,pair_score=[],[]
        new_a1,new_a2,a1_score,a2_score=deal_span(span_data[i])
        if len(new_a1)>0 and len(new_a2)>0:
            for h in range(len(new_a1)):
                for j in range(len(new_a2)):
                    a1_list = list(range(int(new_a1[h][0]),int(new_a1[h][1])))
                    a2_list = list(range(int(new_a2[j][0]),int(new_a2[j][1])))
                    temp_list = [f for f in a1_list if f in a2_list]
                    if len(temp_list)==0:
                        temp_tags=torch.zeros((len(testfeature[i]['token_id'])-2))    #-2是因为原来的token_map考虑了cls和sep
                        temp_tags[int(new_a1[h][0])]=1
                        temp_tags[int(new_a1[h][0])+1:int(new_a1[h][-1])]=2
                        temp_tags[int(new_a2[j][0])]=3
                        temp_tags[int(new_a2[j][0])+1:int(new_a2[j][-1])]=4
                        tags.append(temp_tags)
                        pair_score.append((a1_score[h]+a2_score[j])/2)
        if len(tags)==0:                                               #不可能让一个句子没有tags啊！所以如果计算是没有符合条件的tags，就把tags都当做0吧
            temp_tags=torch.zeros((len(testfeature[i]['token_id'])-2))                #-2是因为原来的token_map考虑了cls和sep
            tags.append(temp_tags)
            pair_score.append(0)
        for num in range(len(tags)):
            tags[num]=tags[num].tolist()
        if args.pos:
            feature={"sents":testfeature[i]['sents'],"input_id":testfeature[i]['input_id'],"sent_tags":tags,"pair_score":pair_score,
                         "token_id":testfeature[i]['token_id'],"pos_id":testfeature[i]['pos_id']}
        else:
            feature={"sents":testfeature[i]['sents'],"input_id":testfeature[i]['input_id'],"sent_tags":tags,"pair_score":pair_score,
                         "token_id":testfeature[i]['token_id']}
        result.append(feature)
    return result


def findindex(templist,num):
    result=[]
    for i in range(len(templist)):
        if templist[i]==num:
            result.append(i)
    return result


def deal_tag(a_list,logit,pair_score,predict=False):
    result=[]
    for i in range(len(a_list)):
        temp_list = a_list[i][1:-1] if predict==True else a_list[i]
        a1=findindex(temp_list,1)+findindex(temp_list,2)
        a2=findindex(temp_list,3)+findindex(temp_list,4)
        pred=findindex(temp_list,5)
        if len(a1)==0 or len(a2)==0 or len(pred)==0:     #约束，主要是遇到空pred的情况，a1和a2倒是好点
            continue
        pred_score=0
        for num in pred:
            pred_score+=logit[i][num][5]
        pred_score = pred_score/len(pred)
        #score = (float(pred_score)+pair_score[i]*2)/3   #这个是用deal_enti_result1的，记录一下
        score = (float(pred_score)+pair_score*2)/3
        feature={'a1':a1,'a2':a2,'pred':pred,'score':score}
        result.append(feature)
    return result


def predict(rel_model,feature_data,device,pos,file_name,carb=False):
    preds,rel_true,logits = [],[],[]
    dataloader = DataLoader(feature_data, batch_size=4, shuffle=False, collate_fn=utils.rel_collate_fn, drop_last=True)
    for batch in dataloader:
        rel_model.eval() 
        if pos:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device),'arg_seq': batch[4].to(device),'seq_len':batch[5],'pos_info':batch[8].to(device),}
        else:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device),'arg_seq': batch[4].to(device),'seq_len':batch[5],}
        
        with torch.no_grad():
            output = rel_model(**inputs)     #产出一个list[list],大长度就是所有文档或句子，里面的长度是token_num
            preds.extend(output[0])          #此时，preds里是每一个句子里的tags，而且可能是很多个tags
            logits.extend(output[1])         #这里用extend还是append还真不确定，一会儿得测试一下,维度最好是n,seq_num,word_num,6
    
    print('Rel_result Done!')
    out_f = open(file_name,'w')
    for num in range(len(preds)):           
        y_pred=preds[num]                    #seq_num,word_num
        pair_score=feature_data[num]['pair_score']
        pred_dic=deal_tag(y_pred,logits[num],pair_score[0],predict=True)   #里面是a1,a2,pred,score的字典,其中前三个都是list,score是float       
        
        #sign = tempdata[num]['sent'] if json else tempdata[num].split(' ')
        if carb:
            sents=' '.join(feature_data[num]['sents'][:-3]) 
        else:
            sents=' '.join(feature_data[num]['sents'])    
        result=[]
        sign=feature_data[num]['sents']
        for i in range(len(pred_dic)):                 #pred_dic为空的时候这个循环压根不会跑，所以没问题
            a1=[sign[j] for j in pred_dic[i]['a1']]
            a2=[sign[j] for j in pred_dic[i]['a2']]
            rel=[sign[j] for j in pred_dic[i]['pred']]
            a1_tags=' '.join(a1)
            a2_tags=' '.join(a2)
            rel=' '.join(rel)
            result.append({'arg1':a1_tags,'rel':rel,'arg2':a2_tags,'score':pred_dic[i]['score']})
            out_f.write(sents+'\t<arg1> '+a1_tags+' </arg1> <rel> '+rel+' </rel> <arg2> '+a2_tags+' </arg2>\t'+str(pred_dic[i]['score'])+'\n')       
    out_f.close()
    return result

def main(args):
    utils.set_seed(10)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with open('datasets/LSOIE/test_all.json','r') as f:      #datasets/LSOIE/dev_all.json
        test_data = json.load(f)
    
    new_test_data=dealdata.deal_conll(test_data)
    test_feature=dealdata.dealdata(new_test_data,8,40,train=False,pos_file='datasets/LSOIE/test_all_pos.txt')
    
    with open(args.span_result_file,'r') as f:
        span_data=f.readlines()
    temp_feature=deal_enti_result(span_data,test_feature,pos=args.pos)
    print('Enti Result Done!!!')
    rel_config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
    model =AutoModel.from_pretrained('microsoft/deberta-v3-base',config=rel_config,)
    
    if args.model_type=='bert_crf_noposnoCLS':   
        rel_model=models.bert_noCLS(rel_config,model,6,False)
    if args.model_type=='bert_crf_noCLS':       
        rel_model=models.bert_noCLS(rel_config,model,6,True)
    if args.model_type=='bert_crf_nopos':        
        rel_model=models.base_model(rel_config,model,6,False)
    if args.model_type=='onlstm_pure':           
        layer_size=[rel_config.hidden_size+5]+[6]+[rel_config.hidden_size+5]
        on_lstm1 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        on_lstm2 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        rel_model=models.bert_bi_onlstm_pure(rel_config,model,6,on_lstm1,on_lstm2,False)
    if args.model_type=='lstm_pure':             
        rel_model=models.bert_bi_lstm_pure(rel_config,model,6,False)
    if args.model_type=='all':                   
        layer_size=[2*rel_config.hidden_size+50]+[6]+[2*rel_config.hidden_size+50]
        on_lstm1 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        on_lstm2 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        rel_model=models.bert_bi_onlstm(rel_config,model,6,on_lstm1,on_lstm2,True)
    if args.model_type=='lstm_all':              
        rel_model=models.bert_bi_lstm(rel_config,model,6,True)                                               
    
    rel_model.load_state_dict(torch.load(args.rel_model_dic))
    rel_model.to(device)                        
    lstm_output=predict(rel_model,temp_feature,device,args.pos,file_name=args.output_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0) 
    parser.add_argument('--model_type', default='lstm_pure', type=str)   
    parser.add_argument('--rel_model_dic', default='SOR-IE', type=str)   
    parser.add_argument('--pos', type=bool, default=False)
    parser.add_argument('--output_files', type=str, default='LSOIE')
    parser.add_argument('--span_result_file', type=str)
    
    #python predict_ablative.py --output_files ../openie6-master/MyORE_test/Mymodel/LSOIE/onlstm_pure_noneg.carb --rel_model_dic model_result/LSOIE/onlstm_pure_noneg.pth --span_result_file ../Triaffine-nested-ner-main/output/LSOIE_final/test_7_predict.txt --pos False --model_type onlstm_pure

    args = parser.parse_args()

    main(args)