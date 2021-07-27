# MEL-GHMF： A Multimodal Entity Linking Model with Gated Hierarchical Fusion

##1. Introduction
MEL-GHMF is a multimodal entity linking method with gated hierarchical multimodal fusion.  First,  it  extracts  hierarchical  textual  andvisual  co-attended  features  to  discover  fine-grained  inter-modalcorrelations by  multimodal  co-attention  mechanisms:  textual-guided  visual  attention  and  visual-guided  textual  attention.  The former   attention   obtains   weighted   visual features  under  the guidance of textual information. In contrast, the latter attention produces weighted textual features under the guidance of visual information. Afterwards, gated fusion is used to adaptively evaluate the importance of hierarchical features of different modalities and  integrate  them  as  the  final  multimodal  representations  of mentions. Finally, the linking entities are selected by calculating the  cosine  similarity  between  representations  of  mentions  andentities in knowledge graph. 


##2. Usage of the code
1. Extract Visual features by the command:    sh img_feat_ex.sh
2. Prepare datasets by the command:              sh nel_data_all.sh
3. Train the model by the command:                     sh nel_train.sh
4. Test the model by the command:                     sh nel_test.sh

##3. Datasets

**Twitter-MEL**:  It is a dataset with more than 31K multimodal samples from Tweets. 
**WikiMEL**: For WikiMEL, we first collect entities from Wikidata, and then extract the textual and visual description of each entity from Wikipedia. WikiMEL has more than 22K multimodal samples.
**Richpedia-MEL**: For Richpedia-MEL, we first collect the Wikidata ids of entities in a large-scale multimodal knowledge graph Richpedia, then obtain the corresponding multimodal information from 524 Wikipedia pages. Richpedia-MEL has more than 17K multimodal 525 samples. 

The KG used in this paper is extracted from Wikidata, which has more than 170K triples and about 80K entities. 

Full datasets and detailed descriptions are accessed via https://github.com/seukgcode/MELBench
