import torch
from CRF.testcrf import CRF
from torch.cuda.amp import autocast as autocast
#BERT,BERT+crf,BERT+POS+crf,bert+lstm+crf,glove_on_lstm_crf,bert+Bi_on_lstm+crf

#基础类,BERT+CRF
class base_model(torch.nn.Module):                                 
    def __init__(self, config, model, rel_label,pos_result):     #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__()   
        self.config = config 
        self.rel_label = rel_label
        self.model = model  
        self.pos_result=pos_result
        if self.pos_result==False:
            self.FNN =  torch.nn.Linear(2*self.config.hidden_size+5, self.rel_label)
        else:
            self.FNN = torch.nn.Linear(2*self.config.hidden_size+5+45, self.rel_label)
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
    
    @autocast()  
    def forward(self,input_id,input_mask,token_map,crf_mask,arg_seq,rel_tags=None,pos_info=None,seq_len=None):

        result_input=self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)   #这里就是bert+CLS+arg_seq+pos
        logits = self.FNN(self.dropout(result_input))
        
        result,loss,logit_result =[],0,[]
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte() 
            temp_result = self.rel_crf.decode(emissions=logits[i],mask=temp_crf_mask)   #seq_num,max_token_len
            logit_result.append(logits[i])
            result.append(temp_result[0:seq_len[i]])                                    #由于seq_num也补齐了，所以这里只取相应的
        
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = logits[i,:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logit_result,)   #这结果应该是list[list[list]]                    
    
    def get_input(self,input_id,input_mask,token_map,arg_seq,crf_mask,pos_info):             
        bert_output= self.model(input_ids=input_id, attention_mask=input_mask)      #bert结果，维度是b,l,768 (l=max(len(input_id)))
        input_mask=input_mask.unsqueeze(-1).repeat(1,1,self.config.hidden_size)
        bert_output = bert_output[0]*input_mask                                     #由于0也会被计算注意力，所以这里要处理一下
        
        token_input = self.get_token_input(bert_output,token_map)                   #把分词的词进行最大池化并补齐
        CLS_embed = self.get_mean_CLS(bert_output,token_map)                        #句子嵌入的表示
        #CLS_embed = bert_output[:,0,]                                              #维度都确认过，没问题,应该是b,768
        
        CLS = CLS_embed.unsqueeze(1).repeat(1,len(token_input[0]),1)
        temp_input = torch.cat((token_input,CLS),-1)                                #嵌入,b,max_token_len,2*768,后面的不管是因为crf的时候会排除
        
        seq_num=crf_mask.size()[1]
        new_temp_input = temp_input.unsqueeze(1).repeat(1,seq_num,1,1).cuda() 
        
        if self.pos_result==False:
            final_input = torch.cat((new_temp_input,arg_seq),-1)
        else:
            final_input = torch.cat((new_temp_input,arg_seq,pos_info),-1)         #b,max_seq_num,max_token_len,2*768+5+45
        
        
        return final_input                                                  #所以这个


    def get_token_input(self,bert_output,token_map):
        max_token_len=max([len(f) for f in token_map])
        token_input=torch.zeros((len(bert_output),max_token_len,self.config.hidden_size))
        for i in range(len(bert_output)):                                           #其实是batch
            index=0
            new_tensor=torch.zeros((max_token_len,self.config.hidden_size))
            for j in range(len(token_map[i])):
                if len(token_map[i][j])==1:
                    new_tensor[index]=bert_output[i][token_map[i][j]]
                if len(token_map[i][j])>1:
                    new_tensor[index]=bert_output[i,token_map[i][j][0]:token_map[i][j][-1]+1,].max(0)[0]
                index += 1
            token_input[i]=new_tensor   
        return token_input
    #这里没有去除SEP和CLS的影响的原因是我在tags里用0补齐了他们的标签，虽然这样会使F1值变得更高一些，但是不会太多
    
    def get_mean_CLS(self,bert_output,token_map):                                    #按实际的token数进行句子的平均池化,并排除了CLS和SEP
        max_token_len=max([len(f) for f in token_map])
        CLS_embed=torch.zeros((len(bert_output),self.config.hidden_size))
        for i in range(len(bert_output)):                 
            token_len=len(token_map[i])
            CLS_embed[i,1:token_len]=bert_output[i,1:token_len,].mean(1)
        return CLS_embed


#测试继承是不是和原来模型一样,测试结果确实一样
class bert_noCLS(base_model):
    def __init__(self, config, model, rel_label, pos_result):        #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__(config, model, rel_label,pos_result)   
        if self.pos_result==False:
            self.FNN =  torch.nn.Linear(self.config.hidden_size+5, self.rel_label)
        else:
            self.FNN = torch.nn.Linear(self.config.hidden_size+5+45, self.rel_label)
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
        
    @autocast()   
    def forward(self,input_id,input_mask,token_map,arg_seq,crf_mask,seq_len,rel_tags=None,pos_info=None):
        temp_logits = self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)  #b,max_seq_num,max_token_len,2*768+5
        logits = self.FNN(self.dropout(temp_logits))
        result,loss,logit_result=[],0,[]
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte() 
            temp_result = self.rel_crf.decode(emissions=logits[i],mask=temp_crf_mask)   #seq_num,max_token_len
            result.append(temp_result[0:seq_len[i]])                                    #由于seq_num也补齐了，所以这里只取相应的
            logit_result.append(logits[i])
            
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = logits[i,:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logit_result,)    #这结果应该是list[list[list]]                    
    
    def get_input(self,input_id,input_mask,token_map,arg_seq,crf_mask,pos_info):             
        bert_output= self.model(input_ids=input_id, attention_mask=input_mask)      #bert结果，维度是b,l,768 (l=max(len(input_id)))
        input_mask=input_mask.unsqueeze(-1).repeat(1,1,self.config.hidden_size)
        bert_output = bert_output[0]*input_mask                                     #由于0也会被计算注意力，所以这里要处理一下
        
        token_input = self.get_token_input(bert_output,token_map)                   #把分词的词进行最大池化并补齐
        seq_num=crf_mask.size()[1]
        new_temp_input = token_input.unsqueeze(1).repeat(1,seq_num,1,1).cuda() 
        
        if self.pos_result==False:
            final_input = torch.cat((new_temp_input,arg_seq),-1)
        else:
            final_input = torch.cat((new_temp_input,arg_seq,pos_info),-1)         #b,max_seq_num,max_token_len,2*768+5+45
            
        return final_input                                                  #所以这个

#双向
class bert_bi_lstm_pure(bert_noCLS):
    def __init__(self, config, model, rel_label,pos_result):        #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__(config, model, rel_label,pos_result)     
        if self.pos_result==False:
            self.lstm = torch.nn.LSTM(config.hidden_size+5,rel_label,2,batch_first=True,bidirectional=True,dropout=0.2)   
        else:
            self.lstm = torch.nn.LSTM(config.hidden_size+50,rel_label,2,batch_first=True,bidirectional=True,dropout=0.2)
        self.FNN = torch.nn.Linear(2*rel_label, rel_label)
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
        
    @autocast()    
    def forward(self,input_id,input_mask,token_map,arg_seq,crf_mask,seq_len,rel_tags=None,pos_info=None):
        temp_logits = self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)
      
        result,loss,logits=[],0,[]
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte() 
            packed_logits=torch.nn.utils.rnn.pack_padded_sequence(temp_logits[i], torch.sum(temp_crf_mask, dim=1).cpu(), batch_first=True, 
                                                                  enforce_sorted=False)
            lstm_output,_ = self.lstm(packed_logits)
            lstm_output2=torch.nn.utils.rnn.pad_packed_sequence(lstm_output,batch_first=True,total_length=temp_logits[i].shape[1])

            output=self.FNN(self.dropout(lstm_output2[0]))    
            
            temp_result = self.rel_crf.decode(emissions=output,mask=temp_crf_mask)  #seq_num,max_token_len
            result.append(temp_result[0:seq_len[i]])                                #由于seq_num也补齐了，所以这里只取相应的
            new_logits = torch.softmax(output,dim=-1)
            logits.append(new_logits[0:seq_len[i]])
            
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = output[:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logits,)    #这结果应该是list[list[list]] 



#双向无pos无cls
class bert_bi_onlstm_pure(bert_noCLS):
    def __init__(self, config, model, rel_label, on_lstm1,on_lstm2,pos_result):        #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__(config, model, rel_label,pos_result)     
        self.on_lstm1 = on_lstm1
        self.on_lstm2 = on_lstm2  
        if self.pos_result==False:
            self.FNN =  torch.nn.Linear(2*(self.config.hidden_size+5), self.rel_label)
        else:
            self.FNN = torch.nn.Linear(2*(self.config.hidden_size+50), self.rel_label)    #如果lstm双向结果是concat那就乘2，不然就单纯加
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
    
    @autocast()    
    def forward(self,input_id,input_mask,token_map,arg_seq,crf_mask,seq_len,rel_tags=None,pos_info=None):
        temp_logits = self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)  #b,max_seq_num,max_token_len,2*768+5
        
        token_len = [len(f) for f in token_map]
        #print(final_input.size(),token_len)
        input2 = torch.zeros_like(temp_logits)                                      
        for i in range(len(crf_mask)):
            for j in range(token_len[i]):
                input2[i,:,j]=temp_logits[i,:,token_len[i]-j-1]                    #这步是倒过来，不然没法反向啊
        
        result,loss,logits_result=[],0,[]
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte()                                      #b,e
            lstm_input = temp_logits[i].permute(1,0,2)                                   #ON_LSTM的输入维度要求是s,b,input_size
            lstm_output = self.on_lstm1(lstm_input,self.on_lstm1.init_hidden(len(lstm_input[0])))[0].permute(1,0,2)
            
            lstm_input2 = input2[i].permute(1,0,2)
            lstm_output2 = self.on_lstm2(lstm_input2,self.on_lstm2.init_hidden(len(lstm_input2[0])))[0].permute(1,0,2)
            
            #logits = self.FNN(self.dropout(lstm_output+lstm_output2))                            #seq_num,max_token_len,2*768+5
            logits = self.FNN(self.dropout(torch.concat((lstm_output,lstm_output2),dim=2)))          #seq_num,max_token_len,2*768+5
            
            temp_result = self.rel_crf.decode(emissions=logits,mask=temp_crf_mask)  #seq_num,max_token_len
            result.append(temp_result[0:seq_len[i]])                               #由于seq_num也补齐了，所以这里只取相应的
            new_logits = torch.softmax(logits,dim=-1)
            logits_result.append(new_logits[0:seq_len[i]])
            
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = logits[:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logits_result,)    #这结果应该是list[list[list]]


class bert_bi_lstm(base_model):          
    def __init__(self, config, model, rel_label,pos_result):        #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__(config, model, rel_label,pos_result)     
        if self.pos_result==False:
            self.lstm = torch.nn.LSTM(2*config.hidden_size+5,rel_label,2,batch_first=True,bidirectional=True,dropout=0.2)   
        else:
            self.lstm = torch.nn.LSTM(2*config.hidden_size+50,rel_label,2,batch_first=True,bidirectional=True,dropout=0.2)
        self.FNN = torch.nn.Linear(2*rel_label, rel_label)
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
        
    @autocast()    
    def forward(self,input_id,input_mask,token_map,arg_seq,crf_mask,seq_len,rel_tags=None,pos_info=None):
        temp_logits = self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)
      
        result,loss,logits=[],0,[]
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte() 
            packed_logits=torch.nn.utils.rnn.pack_padded_sequence(temp_logits[i], torch.sum(temp_crf_mask, dim=1).cpu(), batch_first=True, 
                                                                  enforce_sorted=False)
            lstm_output,_ = self.lstm(packed_logits)
            lstm_output2=torch.nn.utils.rnn.pad_packed_sequence(lstm_output,batch_first=True,total_length=temp_logits[i].shape[1])

            output=self.FNN(self.dropout(lstm_output2[0]))    
            
            temp_result = self.rel_crf.decode(emissions=output,mask=temp_crf_mask)  #seq_num,max_token_len
            result.append(temp_result[0:seq_len[i]])                                #由于seq_num也补齐了，所以这里只取相应的
            new_logits = torch.softmax(output,dim=-1)
            logits.append(new_logits[0:seq_len[i]])
            
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = output[:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logits,)    #这结果应该是list[list[list]] 


class enti_bert_softmax(torch.nn.Module):
    def __init__(self, config, model,num_labels):
        super().__init__()   
        self.config = config
        self.model = model    
        self.enti_FNN = torch.nn.Linear(2*config.hidden_size+100, num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.size_embeddings = torch.nn.Embedding(30, 100)                             #这个20的意思是对0到100个数进行嵌入,后面是维度
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')                        #就是crossentryloss类
        
    def forward(self,input_id,input_mask,enti_mask,enti_size,crf_mask,enti_tags=None):
                                                                                      
        bert_output= self.model(input_ids=input_id, attention_mask=input_mask,)        
        bert_output = bert_output[0]                                                   #bert结果，维度是b,l,768 (l=max(len(input_id)))
        size_embeddings = self.size_embeddings(enti_size)                              #维度是b,enti_num,768(enti_size维度是b,enti_num)
        
        if enti_tags==None:
            enti_result = self.enti_consert(bert_output,size_embeddings,enti_mask,crf_mask)
            output = (enti_result,) + (crf_mask,)                                      #留下crf_mask是为了下一步做准备
        else:
            enti_result, enti_loss = self.enti_consert(bert_output,size_embeddings,enti_mask,crf_mask,enti_tags)
            output = (enti_loss,) + (enti_result,) 
        
        return output
        
    def enti_consert(self,bert_output,size_embeddings,enti_mask,crf_mask,tags=None):
        m = (enti_mask.unsqueeze(-1) == 0).float() * (-1e10)                           #维度为b,enti_num,l,1,且mask的部分为负无穷
        entity_spans_pool = m + bert_output.unsqueeze(1).repeat(1, enti_mask.shape[1], 1, 1)  
        entity_spans_pool = entity_spans_pool.max(dim=2)[0]                            #维度为b,enti_num,768
        
        CLS_embed = bert_output[:,0].unsqueeze(1).repeat(1,enti_mask.shape[1],1)    
        entity_rep=torch.cat([entity_spans_pool, size_embeddings,CLS_embed],dim=2)
        
        logits = self.enti_FNN(self.dropout(entity_rep))                               #维度为b,e,label_num
        
        if tags is not None:
            enti_logits = logits.view(-1,logits.shape[-1])
            enti_tags = tags.view(-1)                                                  #tags应该是个单独的向量，里面是0,1,2的东西
            enti_loss = self.loss(enti_logits,enti_tags)           
            new_mask = crf_mask.view(-1).float()
            loss = (enti_loss * new_mask).sum()/new_mask.sum()                         #这个是按照spert的模板做的                                                                            
            
            return logits, loss                          
        else:
            new_logits = torch.softmax(logits,dim=2)                                   #这个维度也为b,e,label_num并且可以当做置信score
            return new_logits
        

#之前的ON_LSTM_all的结果
class bert_bi_onlstm(base_model):
    def __init__(self, config, model, rel_label, on_lstm1,on_lstm2,pos_result):        #arg_seq=5,对应a1-B,a1-I,a2-B和a2-I和0
        super().__init__(config, model, rel_label,pos_result)     
        self.on_lstm1 = on_lstm1
        self.on_lstm2 = on_lstm2  
        if self.pos_result==False:
            self.FNN =  torch.nn.Linear(4*self.config.hidden_size+20, self.rel_label)
        else:
            self.FNN = torch.nn.Linear(4*self.config.hidden_size+100, self.rel_label)       #这里是concat的FNN
        self.dropout = torch.nn.Dropout(0.1)
        self.rel_crf = CRF(num_tags=rel_label,batch_first=True)
        
    @autocast()    
    def forward(self,input_id,input_mask,token_map,arg_seq,crf_mask,seq_len,rel_tags=None,pos_info=None):
        temp_logits = self.get_input(input_id,input_mask,token_map,arg_seq,crf_mask,pos_info)  #b,max_seq_num,max_token_len,2*768+5
        
        token_len = [len(f) for f in token_map]
        #print(final_input.size(),token_len)
        input2 = torch.zeros_like(temp_logits)                                      
        for i in range(len(crf_mask)):
            for j in range(token_len[i]):
                input2[i,:,j]=temp_logits[i,:,token_len[i]-j-1]                    #这步是倒过来，不然没法反向啊
        
        result,logits_result,loss=[],[],0
        for i in range(len(crf_mask)):
            temp_crf_mask = crf_mask[i].byte()                                      #b,e
            lstm_input = temp_logits[i].permute(1,0,2)                                   #ON_LSTM的输入维度要求是s,b,input_size
            lstm_output = self.on_lstm1(lstm_input,self.on_lstm1.init_hidden(len(lstm_input[0])))[0].permute(1,0,2)
            
            lstm_input2 = input2[i].permute(1,0,2)
            lstm_output2 = self.on_lstm2(lstm_input2,self.on_lstm2.init_hidden(len(lstm_input2[0])))[0].permute(1,0,2)

            #logits = self.FNN(self.dropout(lstm_output+lstm_output2))                            #seq_num,max_token_len,2*768+5
            logits = self.FNN(self.dropout(torch.concat((lstm_output,lstm_output2),dim=2)))          
            #logits = logits.float()                                                              #防止混合精度时传入nan，所以手动改变他的类型
            temp_result = self.rel_crf.decode(emissions=logits,mask=temp_crf_mask)               #seq_num,max_token_len
            result.append(temp_result[0:seq_len[i]])                                             #由于seq_num也补齐了，所以这里只取相应的
            new_logits = torch.softmax(logits,dim=-1)
            logits_result.append(new_logits[0:seq_len[i]])
            
            if rel_tags != None:
                mask = temp_crf_mask[:seq_len[i],]
                logit = logits[:seq_len[i],]
                tags = rel_tags[i,:seq_len[i],]              
                temp_loss = - self.rel_crf(emissions=logit,tags=tags,mask=mask,reduction='token_mean')
                loss += temp_loss
        
        if rel_tags !=None:
            return (loss,) + (result,)
        else:
            return (result,) + (logits_result,)  #这结果应该是list[list[list]]

