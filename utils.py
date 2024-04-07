import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AdamW, get_linear_schedule_with_warmup
import numpy as np
from apex import amp     #这个apex库和我得torch版本不兼容，后来改了一下才能继续用
import random

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if  torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def neg_data(temp_data):
    result=[]
    for data in temp_data:
        new_a1,new_a2 = [],[]
        for i in range(len(data['a1_tags'])):
            if data['a1_tags'][i] not in new_a1:
                new_a1.append(data['a1_tags'][i])
            if data['a2_tags'][i] not in new_a2:
                new_a2.append(data['a2_tags'][i])
        if len(new_a1)==1 or len(new_a2)==1:
            feature={"sents":data['sents'],"input_id":data['input_id'],"sent_tags":data['sent_tags'],
                     "token_id":data['token_id'],"pos_id":data['pos_id']}
            result.append(feature)
            continue
        #上面就是筛选完a1和a2后，发现构造不出负关系的

        #能构造负关系的就全构造负关系，然后和正关系做对比，如果正关系的a1和a2都对上了，那就不要负关系，就这么干！
        tags=[]
        for i in range(len(new_a1)):
            for j in range(len(new_a2)):
                temp_tags=torch.zeros(len(data['sent_tags'][0]))
                a1_list = list(range(new_a1[i][0],new_a1[i][1]))
                a2_list = list(range(new_a2[j][0],new_a2[j][1]))
                temp_list = [f for f in a1_list if f in a2_list]
                if len(temp_list)==0:
                    temp_tags[new_a1[i][0]]=1
                    temp_tags[new_a1[i][0]+1:new_a1[i][-1]]=2
                    temp_tags[new_a2[j][0]]=3
                    temp_tags[new_a2[j][0]+1:new_a2[j][-1]]=4
                    tags.append(temp_tags)
        #这样所有可能的关系全构造完毕，然后考虑和正关系的关系问题
        for h in range(len(data['sent_tags'])):
            sent_tag = torch.tensor(data['sent_tags'][h])
            pos_tag = sent_tag*((sent_tag<5).float())
            for j in range(len(tags)):
                temp_tag = tags[j]
                if (pos_tag == temp_tag).sum()==len(pos_tag):         #即每个位置都相同
                    tags[j]=sent_tag
                    continue
        for num in range(len(tags)):
            tags[num]=tags[num].tolist()
        #这样基本就构造完了，这个tags可以直接替代
        feature={"sents":data['sents'],"input_id":data['input_id'],"sent_tags":tags,
                 "token_id":data['token_id'],"pos_id":data['pos_id']}
        #因为关系模型就需要这么几个东西，所以别的就不要了，sent_id和sents是留着debug时做标识用的
        result.append(feature)
    return result
#测试通过

def tag_pad(rel_tags,seq_num,max_token_len):
    result = torch.zeros((len(rel_tags),max(seq_num),max_token_len),dtype=torch.long)
    for i in range(len(rel_tags)):
        temp_tensor=torch.tensor(rel_tags[i])
        result[i,:temp_tensor.size()[0],:temp_tensor.size()[-1]]=temp_tensor
    return result

def seq_pad(new_arg_seq,max_token_len):
    max_seq_num=max([len(f) for f in new_arg_seq])
    final_seq=[]
    for i in new_arg_seq:
        x=max_seq_num-i.size()[0]
        if x>0:
            temp_tensor=torch.zeros((x,len(i[0])))
            i=torch.cat((i,temp_tensor),0)
        final_seq.append(i)
    
    for i in range(len(final_seq)):
        if i==0:
            result=final_seq[i].unsqueeze(0)
        else:
            temp_tensor=final_seq[i].unsqueeze(0)
            result=torch.cat((result,temp_tensor),0)
    result2=result.unsqueeze(3).long()
    test_tensor=torch.zeros(len(new_arg_seq),max_seq_num,max_token_len,5).scatter(3,result2,1)
    #这不相当于变one_hot的操作
    return test_tensor,max_seq_num                          #token长度和seq_num都padding后的tensor

def rel_collate_fn(batch):
    rel_tags=[]
    new_arg=[]
    for i in batch:
        tags=i['sent_tags']
        new_tags=[]
        for j in tags:
            new_tags.append([0.0]+j+[0.0])
        new_tags=torch.tensor(new_tags)
        arg_tags=new_tags*(1-(new_tags>4).float())     #相当于把5和6mask掉了
        new_arg.append(arg_tags)
        rel_tags.append(new_tags)
        #以上部分是我直接拿数据来做进行测试的
          
    max_len = max([len(f["input_id"]) for f in batch])
    max_token_len = max([len(f['token_id']) for f in batch])
    token_map = [f['token_id'] for f in batch]
    input_ids = [f["input_id"] + [0.0] * (max_len - len(f["input_id"])) for f in batch]          #b,l
    input_mask = [[1.0] * len(f["input_id"]) + [0.0] * (max_len - len(f["input_id"])) for f in batch]
    crf_mask = [[1.0] * (len(f['token_id'])) + [0.0] * (max_token_len - len(f['token_id'])) for f in batch] 
    crf_mask = torch.tensor(crf_mask,dtype=torch.long)
    pos_tensor=torch.zeros(len(batch),max_token_len,45)     #b,max_token,pos
    for i in range(len(batch)):
        temp_pos=[0]+batch[i]['pos_id']+[0]                 #CLS和SEP
        pos_id=torch.tensor(temp_pos).unsqueeze(1).long()
        pos_tensor[i].scatter_(1,pos_id,1)                  
    #这个是得到了batch的pos矩阵，维度是b,max_token,pos,接下来只要变成b,seq_num,max_token,pos就行了
    
    arg=[]
    for i in range(len(new_arg)):
        if new_arg[i].size()[-1]<max_token_len:
            temp_tensor=torch.zeros((new_arg[i].size()[0],max_token_len-new_arg[i].size()[-1]))
            new_tensor=torch.cat((new_arg[i],temp_tensor),-1)
            arg.append(new_tensor)
        else:
            arg.append(new_arg[i])
                     
    padding_result, max_seq_num=seq_pad(arg,max_token_len)   #在这步之前要把arg的长度padding上
    
    seq_len=[len(f) for f in new_arg]                       
    padding_tags = tag_pad(rel_tags,seq_len,max_token_len)
    #这里对rel的操作也是针对原有数据的
    
    crf_mask=crf_mask.unsqueeze(1).repeat(1,max_seq_num,1)  #batch_size,seq_num,token_num
   
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    pos_result = pos_tensor.unsqueeze(1).repeat(1,max_seq_num,1,1)

    output = (input_ids, input_mask, token_map, crf_mask, padding_result, seq_len ,padding_tags, rel_tags,pos_result)
    return output
