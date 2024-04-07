# SOP-IE
这是北京师范大学系统科学学院18级博士生毕业论文工作的一部分，该工作使用[Triaffine-nested-ner](https://github.com/GanjinZero/Triaffine-nested-ner)作为实体识别模型，BERT-LSTM-CRF结构作为关系判断模型，形成一个管道式的开放信息抽取系统。我们的数据集来自[LSOIE](https://github.com/Jacobsolawetz/large-scale-oie)与[CaRB](https://github.com/dair-iitd/CaRB)，为了得到LSOIE的训练数据，请输入以下代码:  
```shell
python get_json_data.py datasets/lsoie_wiki_train.conll datasets/LSOIE/train.json
```
对于CaRB数据，请运行deal_CaRB.ipynb文件即可  

接下来就是实体识别模型与关系判断模型的训练等操作，对于实体识别模型，我们需要修改Triaffine-nested-ner-main中的``train_parser.py``文件的version输入、word、char、pos的pad_id，这些都是根据其中的``word_embed.py``文件得到的；还需要修改``data_util.py``文件中的type和``model\text_encoder.py``文件中的embedding维度。为了得到结果，我们按照如下的代码进行  
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --version LSOIE --model SpanAttModelV3 --bert_name_or_path ../pretraining-models/bert-large-cased/ --learning_rate 1e-5 --batch_size 8 --gradient_accumulation_steps 2 --train_epoch 20 --score tri_affine --truncate_length 100 --word --word_dp 0.2 --word_embed cc --char --pos --warmup_ratio 0.0 --att_dim 256  --bert_before_lstm --lstm_dim 1024 --lstm_layer 2 --encoder_learning_rate 1e-4 --max_span_count 30 --share_parser --subword_aggr max --init_std 1e-2 --dp 0.1   
```
对于关系判断模型，我们使用pytorch的分布式训练，可以按照以下代码来运行，其中N为显卡数量，CaRB参数代表是否是CaRB数据格式，数据文件的路径已经在``DDP_train.py``文件中，可以根据自己需求来修改  
```shell
python -m torch.distributed.launch --nproc_per_node N --use_env DDP_train.py --epochs 15 --world-size N --model_type lstm_pure --batch_size 8 --lr 2e-4  --CaRB False  
```
模型预测请运行代码  
```shell
python predict.py --carb False --input_files test/LSOIE_sentences.txt --output_files result/LSOIE/result_7.carb --rel_model_dic model_result/LSOIE/lstm_pure.pth --span_result_file Triaffine-nested-ner-main/output/LSOIE/test_7_predict.txt 
```
值得注意的是，Triaffine-nested-ner-main结果中的最优epoch在整体结果中并不一定最优，根据观察，在最优epoch的后两轮epoch的结果得到整体最优的结果的可能性最大
