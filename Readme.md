# Semantic Analysis with Embeddings of Austrian Political Party Programms (Grundsatzprogramm)

This Script compares the semantic similarity of the party programs of all big Austrian political partys. 
From the programms, stored in "Dokumente", an embedding is calculated and stored in Chroma DB. 

This is done with

``embed_dokumente.py``

In a second step with

``distance_embeddings.py``

a semantic similarity analysis is done, by several steps. This will produce the graphics in section results.



## Results

### Cosinus-Distanzmatrix

The average of the embedding vector per party is calculated. Then the cosine distance matrix. 
The smaller the value the smaller the semantic distance between these pairs.

|       |FPÖ        | Grüne    |NEOS     |SPÖ      |ÖVP    |
|-------|-----------|----------|---------|---------|-------|    
|FPÖ    | 0.0000    | 0.0401   | 0.0460  | 0.0420  | 0.0384|
|Grüne  | 0.0401    | 0.0000   | 0.0221  | 0.0104  | 0.0226|
|NEOS   | 0.0460    | 0.0221   | 0.0000  | 0.0305  | 0.0240|
|SPÖ    | 0.0420    | 0.0104   | 0.0305  | 0.0000  | 0.0192|
|ÖVP    | 0.0384    | 0.0226   | 0.0240  | 0.0192  | 0.0000|

## Dimension Reduction

 Dimension reduction on the embedings is applied. This allows to express the similarity of the texts visually, by reducing the dimensionallity of the embeding vectors. The closer the observations the more similar are these observations.
 
  Two methods are used for comparison. Multidimensional scaling (MDS) and PCA. It is necessary to keep in mind that a position of the observation points left/right or up/down does not indicate a political position in the left/right spectrum of the partys. MDS and PCA have in terms of position of the points a random element and are not aware of political categorys, the only factor that matters is the distance of the observations with respect to each other. So the semantic similiarity of the programms to each other can be visualised. Both methods produce, in terms of similiar position of the observations to each other a somewhat similiar output.

![alt text](mds_output.png)

![alt text](pca_output.png)



# TODO

Test different Chunksizes and embedding models to have an indication for stability of the output.
Used embeding at the moment is nomic-embed-text with chunk_size=1000 and chunk_overlap=200.