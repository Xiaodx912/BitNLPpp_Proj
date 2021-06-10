# 实验报告  

-------------------------

## 目的   
利用BiLSTM和预训练词向量完成命名实体识别的BIO多分类任务。  

-------------
## 环境与工具  
```text  
python 3.8.8  
pytorch 1.8.1  
numpy  
seaborn  
```  
--------------
## 源代码结构  
> 部分未在下文提到的部分为实验一相关的代码，为版本管理便利未移除。
 - `/bilstm.py`  
   调用工具类完成数据加载、数据格式化、模型训练与模型评估。  
 - `/utils`   
   工具类与工具函数  
   - `/utils/DataAgent.py`  
     文本数据加载、清洗、管理与生成模块。  
     - `class DataAgent`  
       从原始数据文件加载带标注的文本，进行初步分割与处理。为下文中`Encoder`批量数据生成提供便利。  
   - `/utils/Encoder.py`  
     文本编码模块
     - `class BIOLabeledPara`  
       可根据修饰过的词标注信息生成BIO标签向量的数据容器类。  
     - `class WordEmbeddingEncoder`  
       编码器核心  
       接收一个DataAgent对象与预训练词向量文件路径来初始化。在初始化时加载词向量的全部内容，转为`numpy.ndarray`并存储在哈希字典树中。  
       当初始化携带参数`mode='lstm'`时，将在初始化时额外生成一个包含DataAgent内出现的全部单词的单词-编号词典。随后`make_emb_layer() para_to_idx() batch_to_packed_idx()`均依赖本词典。  
       `make_emb_layer()`返回一个`alphabet_size * vector_length`大小的二维`np.ndarray`。可以在转换为`torch.Tensor`后直接用于网络Embedding层初始化。  
       `para_to_idx()`提供了单个段落的编号化编码，返回编码完成的数组和`BIOLabeledPara`。而`batch_to_packed_idx()`调用前者实现了批量编码，返回`PackedSequence`和`List[BIOLabeledPara]`。  
   - `/utils/util_func.py`  
     用于简化主程序的工具性函数。  
     - `calc_masked_result(packed_vec, bio_labels, network: torch.nn.Module, device)`  
       传入`PackedSequence`和`List[BIOLabeledPara]`，并传入网络对象和训练设备，返回不包含padding的网络输出和真实标签，便于计算loss和评估结果。  
 - `/plot.py`  
   加载列表内训练日志并绘制折线图的简易绘图程序。  
 - `/data`  
   用于存放原始文本数据的文件夹。  
 - `/model`  
   用于存放训练模型参数和训练日志的文件夹。  
   
--------------------
## 算法与参数选择   
### 运算方式   
BiLSTM在cpu上缓慢的训练速度让cuda加速成为一种必须。   
cuda加速下的反向传播速度相对采用cpu时有巨大的提升。因此选用cuda加速的训练方式。  

### 网络结构
本次实验中采用了`Embedding(1,50)->BidirectionalLSTM(50,64)->Linear(128,3)`的网络结构。  
`emb`层采用预训练的词嵌入初始化，并设置`requires_grad=False`禁止反向传播。本层将一维整数单词编号映射为50维向量。  
`lstm`层为输入维50，特征数64的双向长短期记忆网络。  
`fc`层为输入维128，输出维3的线性全连接层。  

### 优化器与学习率  
本次实验依然选用了Adam优化器。
本实验使用`ReduceLROnPlateau`来实现学习率的动态调节。  

### 数据加载模式  
RNN的特征导致loss占用空间极大，因此本实验中采用mini-batch分批按权重反向传播的方式。因此数据加载模式为每个epoch划分为若干mini-batch，每次加载一个mini-batch进入显存进行处理。  

---------------
## 实验结果分析  
> #### 关于指标  
> - 宏平均 Macro-average  
>   将BIO三分类视为三个二分类，分别计算F1后平均所得。  
> - 微平均 Micro-average  
>   累加三个分类中True-Positive False-Positive False-Negative后计算的F1值。  
>   因数据中O类占绝大多数，因此一个无视输入并输出O类的分类器也可获得极高的Micro-F1值，因此较为接近1。  
> - BO_F1  
>   为便于与前述实验结果进行对比，将BI分类合并后计算的BO二分类F1值。

本次实验中共计进行了2000轮训练。下图分别为线性和对数横坐标的参数随训练轮次变化折线图。  
![训练过程](fig.png)  
![对数横坐标的训练过程](fig_log.png)  
可以观察到，在训练过程前段，F1在不同频率的波动中稳定上升。  
进入稳定期后，可以观察到数次表现突然恶化的过程，推测可能与陷入局部最优解有关。  
在稳定期可观察到F1较峰值有轻微下降，推测为模型过拟合所致。  

----------------
## 结论  
本次实验的各项评价指标均优于此前实验。  
BiLSTM的原理适于解决输入为不定长文本的NLP问题，本实验初步验证了这种优越性。但是本模型仍有改进的空间。  