# MEL-GHMF： A Multimodal Entity Linking Model with Gated Hierarchical Fusion

## 1. Introduction
MEL-GHMF is a multimodal entity linking method with gated hierarchical multimodal fusion.  First,  it  extracts  hierarchical  textual  andvisual  co-attended  features  to  discover  fine-grained  inter-modalcorrelations by  multimodal  co-attention  mechanisms:  textual-guided  visual  attention  and  visual-guided  textual  attention.  The former   attention   obtains   weighted   visual features  under  the guidance of textual information. In contrast, the latter attention produces weighted textual features under the guidance of visual information. Afterwards, gated fusion is used to adaptively evaluate the importance of hierarchical features of different modalities and  integrate  them  as  the  final  multimodal  representations  of mentions. Finally, the linking entities are selected by calculating the  cosine  similarity  between  representations  of  mentions  andentities in knowledge graph. 

## 2. Principle of MEL-GHMF  
MEL-GHMF includes three steps: (1)  multimodal features extraction, (2) multimodal co-attention, and (3) gated hierarchical  multimodal fusion. 
 First, we respectively extract visual features by ResNet and embed token-level textual features with BERT. 
 Following previous work \cite{lu2016hierarchical}, we convolve phrase embedding vectors using filters with different window sizes, and apply max-pooling across the various n-grams to obtain a single phrase-level representation.
 Afterwards, hierarchical multimodal co-attention mechanism alternately focuses on multimodal information, constructing visual-guided textual features and textual-guided visual features at two levels: token-level and phrase-level.
 We separately combine the guided two-level features of textual and visual information as their respective representations. 
 Gated fusion is then used to adaptively integrate the above representations of textual modality and visual modality into a joint multimodal representation.
 Finally, we obtain the linking entities by measuring the similarity between features of the multimodal representations and candidate entities, which are selected in advance by calculating the standardized edit distance between names of mentions and  entities.

## 3. Usage of the code
1. Extract Visual features by the command:    sh img_feat_ex.sh
2. Prepare datasets by the command:              sh nel_data_all.sh
3. Train the model by the command:                     sh nel_train.sh
4. Test the model by the command:                     sh nel_test.sh

## 4. Datasets

**Twitter-MEL**:  It is a dataset with more than 31K multimodal samples from Tweets. 

**WikiMEL**: For WikiMEL, we first collect entities from Wikidata, and then extract the textual and visual description of each entity from Wikipedia. WikiMEL has more than 22K multimodal samples.

**Richpedia-MEL**: For Richpedia-MEL, we first collect the Wikidata ids of entities in a large-scale multimodal knowledge graph Richpedia, then obtain the corresponding multimodal information from 524 Wikipedia pages. Richpedia-MEL has more than 17K multimodal 525 samples. 

The KG used in this paper is extracted from Wikidata, which has more than 170K triples and about 80K entities. 

Note: In this repository, we only provide part of samples. Full datasets and detailed descriptions can be accessed via our another repository https://github.com/seukgcode/MELBench.
