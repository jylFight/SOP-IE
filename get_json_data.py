#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import json
import random
import csv

#主要是筛选掉一些，头尾关系相同，但关系不同的句子，这些标注我认为是无效标注
def filter_data(temp_data):
    filter_sent = []
    for data in temp_data:
        for i in range(len(data['a1_tags'])):
            for j in range(len(data['a1_tags'])):
                if not i==j:
                    a1=data['a1_tags'][i]
                    a2=data['a2_tags'][i]
                    new_a1=data['a1_tags'][j]
                    new_a2=data['a2_tags'][j]
                    if a1==new_a1 and a2==new_a2 and data['sent_id'] not in filter_sent:
                            filter_sent.append(data['sent_id'])
    result=[]
    for data in temp_data:
        if data['sent_id'] in filter_sent:
            continue
        result.append(data)
    return result,len(filter_sent)

################下面是RE-OIE2016的数据处理################

#这个函数直接就包含了筛选，包括arg长度的筛选和必须有三元组的筛选条件
def get_REOIE2016data(file_path):
    with open(file_path,'r') as f:
        temp_data=json.load(f)             #得到数据，是一个字典
    key_list,value_list = [],[]
    for key in temp_data.keys():
        key_list.append(key)
    for value in temp_data.values():       #这样句子和里面的信息就有了，然后我想测试一下
        value_list.append(value)
    
    result=[]
    for i in range(len(key_list)):
        sents=key_list[i].split(' ')
        a1_tags,a2_tags,tags=[],[],[]                          
        for info in value_list[i]:          #有可能不止一个
            a1=info['arg0_index']
            if a1==[]:                      #先判断有没有
                continue
            a1[-1]=a1[-1]+1
            a2=info['arg1_index']
            if a2==[]:
                continue
            a2[-1]=a2[-1]+1
            if len(info['pred_index'])==3:
                pred=info['pred_index'][1:]
            if len(info['pred_index'])==2:
                if -1 in info['pred_index']:
                    continue
                else:
                    pred=info['pred_index']
            pred[-1]=pred[-1]+1

            a1_len=a1[1]-a1[0]
            a2_len=a2[1]-a2[0]
            if a1_len>12 or a2_len>12:       #arg长度太长认为不是概念，这里设置为12
                continue
            
            sent_tag=torch.zeros(len(sents))
            sent_tag[a1[0]]=1
            sent_tag[a1[0]+1:a1[-1]]=2
            sent_tag[a2[0]]=3
            sent_tag[a2[0]+1:a2[-1]]=4
            sent_tag[pred[0]:pred[-1]]=5

            a1_tags.append(a1)
            a2_tags.append(a2)
            tags.append(sent_tag.tolist())
            
        if len(tags)==0:
            continue
        feature={'sent':sents,'a1_tags':a1_tags,'a2_tags':a2_tags,'sent_id':i,'tags':tags}
        result.append(feature)
    return result

#################OIE-2016的数据，先不管了，数据质量有点过于离谱##################

def get_ORE2016data(file_path):
    f=open(file_path,'r',encoding='utf-8')
    tempdata=[]
    for line in f.readlines():
        tempdata.append(line.split('\t'))

    sent,a1,a2,a1_arg,a2_arg,sents,temp_rel,tags=[],[],[],[],[],[],[],[]
    for data in tempdata:
        if data[0]=='word_id':
            continue
        if data[0]=='\n':                            #句子结束的标志
            a1_size = 0 if a1_arg==[] else a1_arg[-1]-a1_arg[0]+1
            a2_size = 0 if a2_arg==[] else a2_arg[-1]-a2_arg[0]+1
            if a1_size==0 or a2_size==0 or a1_size>8 or a2_size>8:   #防止有的句子没有头尾概念或太长了
                sent,a1_arg,a2_arg,temp_rel=[],[],[],[]
                continue
            tag=torch.zeros(len(sent)) 
        #if len(a1_arg)>0:     #这儿前面应该已经限制了，这里不应该需要判断了啊
            tag[a1_arg[0]]=1
            tag[a1_arg[0]+1:a1_arg[-1]+1]=2
        #if len(a2_arg)>0:
            tag[a2_arg[0]]=3
            tag[a2_arg[0]+1:a2_arg[-1]+1]=4
        #if len(temp_rel)>0:
            tag[temp_rel[0]:temp_rel[-1]+1]=5
            tag = tag.tolist()
            new_tag = [tag]
            #只要句子完事，就得有这个tag
            if sent not in sents:         
                sents.append(sent)
                tags.append(new_tag)
                a1_arg.append(a1_arg[-1]+1)
                a2_arg.append(a2_arg[-1]+1)
                a1.append([a1_arg])
                a2.append([a2_arg])
                sent,a1_arg,a2_arg,temp_rel=[],[],[],[]
                continue
            if sent in sents and sent==sents[-1]:    #这里可以有效把后面的重复句子剔除
                tags[-1].extend(new_tag)
                a1_arg.append(a1_arg[-1]+1)
                a2_arg.append(a2_arg[-1]+1)
                a1[-1].extend([a1_arg])
                a2[-1].extend([a2_arg])
            sent,a1_arg,a2_arg,temp_rel=[],[],[],[]
            continue
            
        if 'A0-B' in data[-1] or 'A0-I' in data[-1]:
            a1_arg.append(int(data[0]))
        if 'A1-B' in data[-1] or 'A1-I' in data[-1]:
            a2_arg.append(int(data[0]))
        if 'P' in data[-1]:
            temp_rel.append(int(data[0]))
        sent.append(data[1])
        
    result=[]
    for num in range(len(sents)):
        feature={'sent':sents[num],'a1_tags':a1[num],'a2_tags':a2[num],'sent_id':num,'tags':tags[num]}
        result.append(feature)
    return result


def writedata(file_path,tags:str):
    x=get_ORE2016data(file_path)
    new_data,num = filter_data(x)
    new_file='datasets/OIE-2016/'+tags+'.json'
    with open(new_file,'w') as f:
        json.dump(new_data,f)
    print(len(x),len(new_data),num)

#测试通过
def main(args):
    assert args.type in ['RE-OIE2016','LSOIE']
    if args.type=='RE-OIE2016':
        file = 'datasets/Re-OIE2016.json'
        test_data = get_REOIE2016data(file)
        for data in test_data:
            if data['sent_id']==341:
                data['sent'][-1]='.'
                data['sent'][-2]='vessel'   #这个数据最后一个词不太对，改一下
    if args.type=='LSOIE':
        test_data = get_ORE2016data(args.input_file)
        
    new_data,num=filter_data(test_data)    
    with open(args.output_file,'w') as f:
        json.dump(new_data_data,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='LSOIE')
    parser.add_argument('--input_file', type=str, default='datasets/lsoie_wiki_train.conll')   
    parser.add_argument('--output_file', type=str, default='LSOIE_train.json')
    args = parser.parse_args()

    main(args)
