import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'     #这样可以指定GPU，注意的是，这个必须在访问GPU之前，即放在最上面
import torch
import json
import torch.distributed as dist
import time
import argparse
import sys
from transformers import AutoConfig, AutoModel
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import warnings
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
warnings.filterwarnings('ignore')

import dealdata
import models
import ON_LSTM
import utils

  #据说不让0卡参与训练只需要加这句就行 
'''
from apex import amp
new_model, optimizer = amp.initialize(rel_model, optimizer, opt_level="O1", verbosity=0)    #据说这儿要在分布式之前
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()
'''
#这里留意一下，需要在命令行里运行这个文件的py文件，直接用ipynb文件不会在环境变量里加东西,未测试
#输入python -m torch.distributed.launch --nproc_per_node 4 --use_env DDP_test.py --epochs 10 --world-size 4 --model_type onlstm_pure --pos False --batch_size 4 --lr 2e-4,node和world_size要一致

#三个卡一起用速度提升了6倍，跑一次用了5个h，主要是因为轮数不用那么多
 
def init_distributed_mode(args):       #变成函数好在main里面调用
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])          #这些环境变量都是用那个命令之后后加的
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        raise EnvironmentError("NOT using distributed mode")

    #print(args.rank,args.world_size,args.gpu)
    args.distributed = True              # 这里需要设定使用的GPU
    torch.cuda.set_device(args.gpu)      # 这里是GPU之间的通信方式，有好几种的，nccl比较快，而且官方推荐nvidia的GPU使用。
    args.dist_backend = 'nccl'            # 启动多GPU， 其中那个method用default值就行
    dist.init_process_group(backend=args.dist_backend,init_method=args.dist_url, world_size=args.world_size,rank=args.rank)
    # 这个是：多GPU之间进行同步，也就是有的GPU跑的快，有的跑的慢（比如当你判断if RANK == 0: do something， 那么进程0就会多执行代码速度慢）
    dist.barrier()                       # 所以这个代码就是等待所有进程运行到此处。


def cleanup():
    # 这里不同我多说，看名字就知道啥意思
    dist.destroy_process_group()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
# 拿到你有几个GPU，数量。主要是用来all_reduce计算的。
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

# 拿到进程的rank
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def reduce_value(value, average=True):
    # 拿到GPU个数，主要是判断我们有几个进程
    world_size = get_world_size()
    # 如果单进程就返回
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)   # 这个就是all_reduce把不同进程的值都汇总返回。
        if average:
            value /= world_size
        return value

def is_main_process():
    return get_rank() == 0

def train_one_epoch(model, optimizer, data_loader, device, pos,enti=False):
    model.train()
    optimizer.zero_grad()
    mean_loss = torch.zeros(1).to(device)
    #scaler = GradScaler()   #主要是测试失败了，就没用，并且性能时间看下来差的不多
    for step, batch in enumerate(data_loader):
        if pos:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],
                    'rel_tags':batch[6].to(device),'pos_info':batch[8].to(device),}
        else:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],
                    'rel_tags':batch[6].to(device)}   
            
        with autocast():
            outputs = model(**inputs)
            loss = outputs[0] 
        
        dist.barrier() 
        loss.backward()                   #有个有意思的事，和半精度比，反向传播时候并没有受到速度的影响，干脆就这样？400秒和408秒的区别而已，完全可以接受
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if step%50==0 and args.rank==0:
            print("num_step:",step,'rank:',args.rank,'loss:',loss.item())

        optimizer.step()
        optimizer.zero_grad()
    # GPU之间同步，
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return loss.item()

def rel_evaluate(model,feature_data,device,pos=True):    #考虑分词的
    preds, rel_true = [], []
    model.eval()
    for batch in feature_data:
        if pos:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],'pos_info':batch[8].to(device),}
        else:
            inputs = {'input_id': batch[0].to(device),'input_mask': batch[1].to(device),'token_map':batch[2],
                      'crf_mask':batch[3].to(device), 'arg_seq': batch[4].to(device), 'seq_len':batch[5],}
        rel_true.extend(batch[7])
        with torch.no_grad():
            with autocast():
                output = model(**inputs)     #产出一个list[list],大长度就是所有文档或句子，里面的长度是token_num
                preds.extend(output[0])
    
    #这里注意一下，要把0项排除才行，这里需要计算一下，可能需要手动计算,这里就直接算micro-f1了
    p_fenmu,r_fenmu,fenzi=0,0,0
    for num in range(len(preds)):                  #这里换一下，只考虑关系词的判定来看看结果
        y_true=(rel_true[num]>4).float()
        y_pred=(torch.tensor(preds[num])>4).float()
        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j]==y_pred[i][j] and y_true[i][j]>0:
                    fenzi+=1
        p_fenmu += ((np.array(y_true))>0).sum()
        r_fenmu += ((np.array(y_pred.cpu()))>0).sum()       
    #这里如果是多进程，要把fenzi、p_fenmu和r_fenmu都加上
    sum_fenzi=reduce_value(torch.tensor(fenzi).to(device),average=False)
    sum_p_fenmu=reduce_value(torch.tensor(p_fenmu).to(device),average=False)
    sum_r_fenmu=reduce_value(torch.tensor(r_fenmu).to(device),average=False)

    return sum_fenzi,sum_p_fenmu,sum_r_fenmu

def get_score(fenzi,p_fenmu,r_fenmu):
    p=float(fenzi/p_fenmu)
    if r_fenmu==0:
        r=0
    else:
        r=float(fenzi/r_fenmu)
    p=round(p,5)
    r=round(r,5)
    if p==0 and r==0:
        result=0
    else:
        result = 2*p*r/(p+r)
    return p,r,result

def main(args):
    start_time=time.time()
    utils.set_seed(10)
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    init_distributed_mode(args)
    rank = args.rank   #获得当前进程的rank
    batch_size = args.batch_size
    args.lr *= args.world_size    #这个需不需要乘也没说明白，据说可以调

    with open('datasets/LSOIE/train_all.json','r') as f:
        train_data = json.load(f)
    with open('datasets/LSOIE/dev_all.json','r') as f:
        dev_data = json.load(f)
    with open('datasets/LSOIE/test_all.json','r') as f:
        test_data = json.load(f)
    
    train_data=dealdata.deal_conll(train_data)
    dev_data=dealdata.deal_conll(dev_data)
    test_data=dealdata.deal_conll(test_data)    
    
    train_feature=dealdata.dealdata(train_data,8,40,train=True,sci=False,pos_file='datasets/LSOIE/train_all_pos.txt')
    dev_feature=dealdata.dealdata(dev_data,8,40,train=False,sci=False,pos_file='datasets/LSOIE/dev_all_pos.txt')
    test_feature=dealdata.dealdata(test_data,8,40,train=False,sci=False,pos_file='datasets/LSOIE/test_all_pos.txt')
    train_neg_feature=utils.neg_data(train_feature)
    
    #train_sample = torch.utils.data.distributed.DistributedSampler(train_neg_feature)
    train_sample = torch.utils.data.distributed.DistributedSampler(train_feature)
    dev_sample = torch.utils.data.distributed.DistributedSampler(dev_feature)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_feature)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sample, args.batch_size, drop_last=True)

    train_dataloader = torch.utils.data.DataLoader(train_feature,batch_sampler=train_batch_sampler,collate_fn=utils.rel_collate_fn, shuffle=False,pin_memory=True)      #注意这里现在是没有负关系的！
    val_dataloader = torch.utils.data.DataLoader(dev_feature,batch_size=args.batch_size, sampler=dev_sample, collate_fn=utils.rel_collate_fn, shuffle=False,pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_feature,batch_size=args.batch_size, sampler=test_sample,collate_fn=utils.rel_collate_fn,  shuffle=False,pin_memory=True)
    device = torch.device(rank)
    
    #print(len(train_feature),len(dev_feature),len(test_feature))
    rel_config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
    model=AutoModel.from_pretrained('microsoft/deberta-v3-base',config=rel_config,)
    if args.model_type=='bert_crf_noposnoCLS':   #验证集P,80.0,R,85.0,F1,82.4,测试集P,80.1,R,84.7,F1,82.3,时间397s/轮
        rel_model=models.bert_noCLS(rel_config,model,6,False)
    if args.model_type=='bert_crf_noCLS':        #验证集P,79.7,R,84.1,F1,81.8,测试集P,79.9,R,84.1,F1,82.0,时间418s/轮
        rel_model=models.bert_noCLS(rel_config,model,6,True)
    if args.model_type=='bert_crf_nopos':        #验证集P,76.6,R,87.2,F1,81.6,测试集P,78.4,R,86.6,F1,82.3,时间384s/轮
        rel_model=models.base_model(rel_config,model,6,False)
    if args.model_type=='onlstm_pure':           #验证集P,85.9,R,88.4,F1,87.2,测试集P,86.4,R,87.9,F1,87.2,时间1795s/轮，neg验证集P,79.3,R,90.1,F1,84.3,测试集P,79.7,R,88.2,F1,83.8,时间1558s/轮,两层neg验证集P,79.7,R,89.4,F1,84.3,测试集P,79.3,R,88.6,F1,83.7
        #这里注意一下，样本程序设定的layer_size是[ninp]+[nhid]*(layres-1)+[ninp],也就是说我在6那里乘2就行了？
        layer_size=[rel_config.hidden_size+5]+[6]*2+[rel_config.hidden_size+5]
        on_lstm1 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        on_lstm2 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        rel_model=models.bert_bi_onlstm_pure(rel_config,model,6,on_lstm1,on_lstm2,False)
    if args.model_type=='lstm_pure':             #验证集P,89.0,R,91.8,F1,90.4,测试集P,88.2,R,91.5,F1,89.8,时间410s/轮，neg
        rel_model=models.bert_bi_lstm_pure(rel_config,model,6,False)
    if args.model_type=='all':                   #验证集P,85.9,R,89.7,F1,87.8,测试集P,85.5,R,89.1,F1,87.3,时间1558s/轮，neg验证集P,78.9,R,92.3,F1,85.1,测试集P,78.8,R,92.1,F1,84.9,时间1619s/轮
        layer_size=[2*rel_config.hidden_size+50]+[6]+[2*rel_config.hidden_size+50]
        on_lstm1 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        on_lstm2 = ON_LSTM.ONLSTMStack(layer_size, chunk_size=1,dropout=0.2, dropconnect=0.2) 
        rel_model=models.bert_bi_onlstm(rel_config,model,6,on_lstm1,on_lstm2,True)
    if args.model_type=='lstm_all':              #neg的结果验证集P,81.5,R,93.1,F1,86.9,测试集P,82.2,R,92.6,F1,87.1,时间469s/轮,no_neg的结果是P,86.4,R,90.3,F1,88.3,时间445s/轮（batch为8*4也就是32）
        rel_model=models.bert_bi_lstm(rel_config,model,6,True)
                                                 #span_model,验证集P,73.5,R,47.1,F1,57.4，测试集P,74.1,R,47.7,F1,58.0,时间187s/轮
    rel_model.to(device)

    optimizer_grouped_parameters = [
        {"params": [p for n, p in rel_model.named_parameters() if 'model' in n ], "weight_decay":0.01 },
        {"params": [p for n, p in rel_model.named_parameters() if not 'model' in n ], "lr": args.lr},
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-6)
    #optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    dist.barrier()
    new_model = torch.nn.parallel.DistributedDataParallel(rel_model,output_device=0, device_ids=[rank],broadcast_buffers=False, find_unused_parameters=True) 
    
    if rank==0:
        print('-----------------Start Training-------------------')
    best_score = -1
    epoch_time=time.time()
    for epoch in range(args.epochs):
        train_sample.set_epoch(epoch)
        mean_loss = train_one_epoch(new_model,optimizer,train_dataloader,device,args.pos)  
        fenzi,p_fenmu,r_fenmu=rel_evaluate(new_model,val_dataloader,device,pos=args.pos)  
        dev_p,dev_r,dev_score = get_score(fenzi,p_fenmu,r_fenmu)
        fenzi2,p_fenmu2,r_fenmu2=rel_evaluate(new_model,test_dataloader,device,pos=args.pos) 
        test_p,test_r,test_score = get_score(fenzi2,p_fenmu2,r_fenmu2)
        epoch_time2=time.time()
        if rank==0:
            if dev_score>best_score:
                best_score=dev_score
                print(epoch,'轮最佳,测试集P为:',test_p,'R为:',test_r,'F1为:',test_score)
                torch.save(new_model.module.state_dict(),'model_result/LSOIE/'+args.model_type+'noneg.pth')   #save不加module，单卡是无法加载模型的
                #这里只保存rank0的原因是所有进程的模型参数是相同的，选择一个就行
            print(epoch,"轮花费时间值为:",epoch_time2-epoch_time,"fenzi为:",int(fenzi),"p_fenmu为:",int(p_fenmu),"r_fenmu为:",int(r_fenmu),'P为:',dev_p,'R为:',dev_r,'F1为:',dev_score)
            
        #这里模型结果我就不存了，也不会存，就直接输出结果吧！不过这里有个问题，就是这个evaluate的过程好像没考虑多进程，得研究一下
    cleanup()
    end_time=time.time()
    if rank==0:
        print('运行时间:',end_time-start_time,'秒')
    return end_time
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)   #只有on_lstm占显存多，其余的batch都可以设置大一点，可以设置为32或者16，但是要相应调整学习率
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--world-size', default=3, type=int, help='number of distributed processes')   #这个据说是用几块GPU的操作
    parser.add_argument('--dist_url', type=str, default='env://')
    parser.add_argument('--pos', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='lstm_pure')
    parser.add_argument('--local_rank', type=int, help='rank of distributed processes')
    args = parser.parse_args()

    main(args)
        