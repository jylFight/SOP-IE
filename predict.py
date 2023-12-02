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


#这个函数可以根据enti_model的结果生成rel的tags
def deal_enti_result(model,testfeature,device):
    result,preds,enti_span,mask,token_id = [],[],[],[],[]
    dataloader = DataLoader(testfeature, batch_size=8, shuffle=False, collate_fn=utils.enti_collate_fn, drop_last=True)
    for num,batch in enumerate(dataloader):
        model.eval()
        inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'enti_mask':batch[4].to(device),
                  'enti_size':batch[3].to(device),'crf_mask': batch[5].to(device),}
        with torch.no_grad():
            output = model(**inputs)     #这个结果不应该是个b,e,label_num的tensor吗？
            preds.extend(output[0])
            mask.extend(batch[5])
            enti_span.extend(batch[7])
            token_id.extend(batch[8])
    
    
    for i in range(len(preds)):              #i代表序号
        y_pred = preds[i].argmax(dim=-1)  
        new_a1,new_a2,a1_score,a2_score,tags,pair_score=[],[],[],[],[],[]    #score就是每个span的概率
        for j in range(int(mask[i].sum())):              #序列
            if y_pred[j]==1:
                new_a1.append(enti_span[i][j])
                a1_score.append(preds[i][j][1])
            if y_pred[j]==2:
                new_a2.append(enti_span[i][j])
                a2_score.append(preds[i][j][2])
            if y_pred[j]==3:
                new_a1.append(enti_span[i][j])
                a1_score.append(preds[i][j][3])
                new_a2.append(enti_span[i][j])
                a2_score.append(preds[i][j][3])
       # if i<10:
        #    print('a1',new_a1,'a2',new_a2)

        if len(new_a1)>0 and len(new_a2)>0:
            for h in range(len(new_a1)):
                for j in range(len(new_a2)):
                    a1_list = list(range(new_a1[h][0],new_a1[h][1]))
                    a2_list = list(range(new_a2[j][0],new_a2[j][1]))
                    temp_list = [f for f in a1_list if f in a2_list]
                    if len(temp_list)==0:
                        temp_tags=torch.zeros((len(token_id[i])-2))    #-2是因为原来的token_map考虑了cls和sep
                        temp_tags[new_a1[h][0]]=1
                        temp_tags[new_a1[h][0]+1:new_a1[h][-1]]=2
                        temp_tags[new_a2[j][0]]=3
                        temp_tags[new_a2[j][0]+1:new_a2[j][-1]]=4
                        tags.append(temp_tags)
                        pair_score.append(float((a1_score[h]+a2_score[j])/2))  #这儿不对，咋干出一堆tags了      
                        
        if len(tags)==0:                                   #不可能让一个句子没有tags啊！所以如果计算是没有符合条件的tags，就把tags都当做0吧
            temp_tags=torch.zeros((len(token_id[i])-2))    #-2是因为原来的token_map考虑了cls和sep
            tags.append(temp_tags)
            pair_score.append(0)
        for num in range(len(tags)):
            tags[num]=tags[num].tolist()  
        if i%50==0:
            print(i)
        feature={"sents":testfeature[i]['sents'],"input_id":testfeature[i]['input_id'],"sent_tags":tags,"pair_score":pair_score,
                 "token_id":testfeature[i]['token_id'],"pos_id":testfeature[i]['pos_id']}
        #因为关系模型就需要这么几个东西，所以别的就不要了，sent_id和sents是留着debug时做标识用的
        result.append(feature)
        print('Enti result Done!')
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
        score = (float(pred_score)+pair_score[i]*2)/3
        feature={'a1':a1,'a2':a2,'pred':pred,'score':score}
        result.append(feature)
    return result


def predict(rel_model,feature_data,device,pos,file_name):
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
        pair_score=temp_feature[num]['pair_score']
        pred_dic=deal_tag(y_pred,logits[num],pair_score,predict=True)   #里面是a1,a2,pred,score的字典,其中前三个都是list,score是float

        #sign = tempdata[num]['sent'] if json else tempdata[num].split(' ')
        sents=' '.join(feature_data[num]['sents'])    #如果是json文件就这个
        result=[]
        sign=feature_data[num]['sents']
        for i in range(len(pred_dic)):
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
    with open(args.input_files,'r') as f:
        test_data=json.load(f)
    test_data = dealdata.deal_conll(test_data) if args.data_type=='LSOIE' else test_data
    if args.data_type=='LSOIE':
        test_feature=dealdata.dealdata(test_data,8,40,train=False,sci=False,pos_file='data/pos/test.txt')
    else:    
        test_feature=dealdata.dealdata(test_data,8,40,train=False,sci=False,pos_file='data/pos/RE_test.txt')
        
    config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
    model = AutoModel.from_pretrained('microsoft/deberta-v3-base',config=config,)
    if args.enti_done:
        with open('data/enti_result.json','r') as f:
            temp_feature=json.load(f)
        print('Enti result Done!')
    else:
        enti_model = models.enti_bert_softmax(config,model,4)
        enti_model.load_state_dict(torch.load('model_result/LSOIE/enti_model.pth'))
        enti_model.to(device)
        temp_feature=deal_enti_result(enti_model,test_feature,device)
    
    model2 =AutoModel.from_pretrained('microsoft/deberta-v3-base',config=config,)
    rel_model=models.bert_bi_lstm_pure(config,model2,6,False)
    rel_model.load_state_dict(torch.load(args.model_dic))
    rel_model.to(device)
    lstm_output=predict(rel_model,temp_feature,device,args.pos,file_name=args.output_files)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1) 
    parser.add_argument('--model_dic', default='SOR-IE', type=str)   
    parser.add_argument('--pos', type=bool, default=False)
    parser.add_argument('--input_files', type=str, default='rel')
    parser.add_argument('--output_files', type=str, default='LSOIE')
    parser.add_argument('--data_type', type=str, default='LSOIE')
    parser.add_argument('--enti_done', type=bool, default=False)
    
    args = parser.parse_args()

    main(args)