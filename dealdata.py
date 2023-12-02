import json
import numpy as np
import torch
from transformers import AutoTokenizer
import random
import string

def get_input(new_sents,tokenizer):
    input_id=[]         #这个是输入，就是分词之后的正常序列，带CLS和SEP的
    token_id=[]         #这个是记录token在分词后的序列里的标记的list，里面是list[list]
    index=0             #用于控制的常数
    for num,word in enumerate(new_sents):
        token_num=[]
        ids=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))    #每个词每个词分
        length=len(ids)
        for i in range(length):
            token_num.append(num+index+i)
        if length>1:
            index += length-1
        input_id.extend(ids)
        token_id.append(token_num)
    return input_id, token_id
#已通过测试

def get_neg(tempdata,max_span_size,neg_enti_num,token_id,input_id,train,old_pos_span=None):
    if old_pos_span==None:
        pos_span=[]
        for i in range(len(tempdata['entities'])):
            enti_start=tempdata['entities'][i]['start']
            enti_end=tempdata['entities'][i]['end']
            pos_span.append([enti_start,enti_end])
    else:
        pos_span = old_pos_span
    
    token_count=len(token_id)-2                       #这里面包含了CLS和SEP
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size)):
            span=[i,i+size]
            if span not in pos_span:
                neg_entity_spans.append(span)
    #上述是把正实体的span排除后得到的所有符合条件的负实体
    if train:
        if len(neg_entity_spans)<neg_enti_num:
            neg_list=neg_entity_spans
        else:
            neg_list=random.sample(neg_entity_spans,neg_enti_num)    #随机取neg_enti_num个 
    else:
        neg_list=neg_entity_spans
    
    index=0
    neg_enti_mask=torch.zeros((len(neg_list),len(input_id)))  
    neg_size=[]
    for neg_tags in neg_list:       
        neg_enti_mask[index,token_id[neg_tags[0]+1][0]:token_id[neg_tags[1]][-1]+1]=1 
        neg_size.append(neg_tags[1]-neg_tags[0])
        index += 1
    enti_span = pos_span + neg_list
    return neg_enti_mask, neg_size, enti_span
#测试通过,span是不考虑cls和sep序号的，但是mask得考虑，因为span是为了方便后面做的

#就比较简单的思路，如果pos_word和当前word不一致，那就只记录头word的信息，然后让pos定位到和下一个word相同的位置
def addpos(temp_data,posfile):
    with open(posfile,'r') as h:            #添加了pos信息的句子
        pos_data = h.readlines()
    with open('pos_dic.json','r') as f:     #pos标签的id和pos对应的字典,注意有个标签是IN/that，得去除一下
        pos_dic=json.load(f)
    
    sent_pos_id,sent_pos_tags=[],[]
    pos_index=0
    for data in temp_data:
        pos_id,pos_tags=[],[]
        for i in range(len(data['sent'])):
            pos_word=pos_data[pos_index].split('\t')[0]
            pos_info=pos_data[pos_index].split('\t')[1]
            if pos_info=='IN/that': pos_info='IN'
            pos_tags.append(pos_info)
            if pos_info not in pos_dic:
                pos_id.append(0)
            else:
                pos_id.append(pos_dic[pos_info])
            pos_index+=1            #控制pos信息表的位置

            if data['sent'][i]!=pos_word:    
            #当前词没匹配上，那我要保证下一个词匹配上就行了，有个弊端，连着两个词是被拆分的话可能会被跳过，索性就一个这样的句子，让我删了嘻嘻！
                if len(pos_tags)==len(data['sent']):
                    continue
                new_word=data['sent'][i+1]
                while new_word!=pos_data[pos_index].split('\t')[0]:
                    pos_index+=1
                    #print(pos_index)
        #if not len(pos_tags)==len(data['sent']):  #这个是测试的，测试通过了
        #    print(i)
        sent_pos_id.append(pos_id)
        sent_pos_tags.append(pos_tags)
    return sent_pos_id,sent_pos_tags

#改一下，头实体1，尾实体2，既是头又是尾的标3，看看准不准
def get_pos_enti(token_id,tempdata,input_id):
    index=0
    enti_mask = torch.zeros(2*len(tempdata['tags']),len(input_id))          #mask,但可能有重复的
    enti_size = []
    enti_tags = []
    head = []
    tail = []
    for i in range(len(tempdata['tags'])):                #由于模型计算的时候以input_id为主，所以这里确实要按照input_id来做
        head_span = tempdata['a1_tags'][i]                #这两个tags都没有考虑cls和sep，所以后面neg的也一样
        tail_span = tempdata['a2_tags'][i]
        if head_span not in head:                         #去重
            head.append(head_span)                        #这里都算上是为了后面筛选既是头又是尾的实体
        if tail_span not in tail:
            tail.append(tail_span)                            
    
    both_span = [x for x in head if x in tail]            #既是头实体又是尾实体
    pure_head = [x for x in head if x not in both_span]
    pure_tail = [x for x in tail if x not in both_span]
    
    for i in pure_head:
        enti_tags.append(1)
        enti_size.append(i[1]-i[0])
        enti_mask[index,token_id[i[0]+1][0]:token_id[i[1]][-1]+1]=1             
        index += 1  
    
    for i in pure_tail:
        enti_tags.append(2)
        enti_size.append(i[1]-i[0])
        enti_mask[index,token_id[i[0]+1][0]:token_id[i[1]][-1]+1]=1
        index += 1  
    
    for i in both_span:
        enti_tags.append(3)
        enti_size.append(i[1]-i[0])
        enti_mask[index,token_id[i[0]+1][0]:token_id[i[1]][-1]+1]=1
        index += 1  
        
    enti_num = len(pure_head) + len(pure_tail) + len(both_span)
    new_enti_mask =  enti_mask[:enti_num,]   
    pos_span = pure_head + pure_tail + both_span
    
    return new_enti_mask,enti_size,enti_tags, pos_span
#测试通过

def deal_conll(tempdata):      #主要是处理OIE数据的a1和a2tags
    features=[]
    for data in tempdata:
        new_a1,new_a2=[],[]
        for a1 in data['a1_tags']:
            #print(a1)
            new_a1.append([a1[0],a1[-1]])
        for a2 in data['a2_tags']:
            new_a2.append([a2[0],a2[-1]])     
        feature={'sent':data['sent'],'a1_tags':new_a1,'a2_tags':new_a2,'tags':data['tags']}
        features.append(feature)
    return features

#这儿得根据LSOIE数据的格式改一份
def dealdata(tempdata,max_span_len,neg_enti_num,train,sci,pos_file=None):
    if sci:
        tokenizer=AutoTokenizer.from_pretrained('KISTI-AI/Scideberta-full',)  
    else:
        tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-base',) 
    
    features=[]
    for data in tempdata:
        sents=data['sent']
        new_sents=['[CLS]']+sents+['[SEP]']
        input_id,token_id = get_input(new_sents,tokenizer) #这里input_id包含CLS和SEP，还有token_id存的是哪些token分词的时候被分解了
            
        pos_enti_mask,pos_enti_size, pos_tag, pos_span = get_pos_enti(token_id,data,input_id)
        #正实体的mask有了，tags也有了，然后看负实体的
        neg_enti_mask,neg_enti_size, enti_span = get_neg(data,max_span_len,neg_enti_num,token_id,input_id,train, pos_span)
        
        #实体信息整合到一起
        enti_mask = torch.cat([pos_enti_mask,neg_enti_mask],0)     
        enti_size = pos_enti_size+neg_enti_size
        final_tags = pos_tag + [0] * len(neg_enti_size)
        
        feature={
            'enti_mask':enti_mask,      
            'enti_size':enti_size,
            'input_id':input_id,
            'enti_tags':final_tags,
            'sent_tags':data['tags'],                 #这里保存原有的tags,且没有cls和sep，因为要进lstm，所以没管
            #'sent_id':data['sent_id'],
            'token_id':token_id,
            'sents':sents,
            'enti_span':enti_span,
            'a1_tags':data['a1_tags'],
            'a2_tags':data['a2_tags'],                #这里a1和a2完全是为了后面方便做
        } 
        features.append(feature)
    if not pos_file==None:
        pos_id,pos_tags = addpos(tempdata,pos_file)           #添加pos信息
        for i in range(len(features)):
            features[i]['pos_id']=pos_id[i]
            features[i]['pos_tags']=pos_tags[i]         
    return(features)