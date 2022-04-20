# MEL-GHMF： A Multimodal Entity Linking Model with Gated Hierarchical Fusion

## 1. Introduction
MEL-GHMF is a multimodal entity linking method with gated hierarchical multimodal fusion.  First,  it  extracts  hierarchical  textual  andvisual  co-attended  features  to  discover  fine-grained  inter-modalcorrelations by  multimodal  co-attention  mechanisms:  textual-guided  visual  attention  and  visual-guided  textual  attention.  The former   attention   obtains   weighted   visual features  under  the guidance of textual information. In contrast, the latter attention produces weighted textual features under the guidance of visual information. Afterwards, gated fusion is used to adaptively evaluate the importance of hierarchical features of different modalities and  integrate  them  as  the  final  multimodal  representations  of mentions. Finally, the linking entities are selected by calculating the  cosine  similarity  between  representations  of  mentions  andentities in knowledge graph. 

## 2. Principle of MEL-GHMF  
MEL-GHMF includes three steps: (1)  multimodal features extraction, (2) multimodal co-attention, and (3) gated hierarchical  multimodal fusion. 
 First, we respectively extract visual features by ResNet and embed token-level textual features with BERT. 
 We convolve phrase embedding vectors using filters with different window sizes, and apply max-pooling across the various n-grams to obtain a single phrase-level representation.
 Afterwards, hierarchical multimodal co-attention mechanism alternately focuses on multimodal information, constructing visual-guided textual features and textual-guided visual features at two levels: token-level and phrase-level.
 We separately combine the guided two-level features of textual and visual information as their respective representations. 
 Gated fusion is then used to adaptively integrate the above representations of textual modality and visual modality into a joint multimodal representation.
 Finally, we obtain the linking entities by measuring the similarity between features of the multimodal representations and candidate entities, which are selected in advance by calculating the standardized edit distance between names of mentions and  entities.
 
![image -w80](https://user-images.githubusercontent.com/18082151/127132229-9612258d-8f36-43a3-af5a-d9300409198a.png)

## 3. Usage of the code
Step1: Extract Visual features by the command:    sh img_feat_ex.sh

Step2: Prepare datasets by the command:              sh nel_data_all.sh

Step3: Train the model by the command:                     sh nel_train.sh

Step4: Test the model by the command:                     sh nel_test.sh


**Hyperparameters**：We set the dimensions of textual and visual features to 512,  MCA  dimension to 512, the number of stacked layers of multimodal co-attention to 2,  mention feature and entity feature dimensions to 768,  heads of MH-SA to 8, dropout to 0.4,  triplet loss interval to 0.5.
We optimize the parameters with AdamW  with batch size 32, learning rate $5\times10^{-5}$,  L2 regularization coefficient 0.2 and gradient clipping threshold 1.0. 

## 4. Datasets

**Twitter-MEL**:  It is a dataset with more than 31K multimodal samples from Tweets. 

**WikiMEL**: For WikiMEL, we first collect entities from Wikidata, and then extract the textual and visual description of each entity from Wikipedia. WikiMEL has more than 22K multimodal samples.

**Richpedia-MEL**: For Richpedia-MEL, we first collect the Wikidata ids of entities in a large-scale multimodal knowledge graph Richpedia, then obtain the corresponding multimodal information from 524 Wikipedia pages. Richpedia-MEL has more than 17K multimodal 525 samples. 

The KG used in this paper is extracted from Wikidata, which has more than 170K triples and about 80K entities. 

The statistics of datasets are summarized in following Table, which contain total samples, mentions, average text length of a sample, and average number of mentions in a sample.

![image](https://user-images.githubusercontent.com/18082151/127133729-f1774f7e-1886-45d2-9844-d228ba07a6b4.png)

Download link of the texts of the three datasets: https://www.aliyundrive.com/s/zyXeLQx7sgh.

Download link of ResNet-200: https://pan.baidu.com/s/1NYxnVn6BJpnGpu7u-7Uy4Q, extract code: 1111.

Note: In this repository, we only provide part of samples. Full datasets and detailed descriptions can be accessed via our another repository https://github.com/seukgcode/MELBench.
