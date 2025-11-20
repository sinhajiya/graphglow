# Region-Aware Makeup Transfer using GCNs

**Members:** [Jiya Sinha](https://github.com/sinhajiya), [Rashi Bharti](https://github.com/rashibharti28), [Sakshi Tiwari](https://github.com/Sakshi-Tiwari2003), [Nandhana KS](https://github.com/nandhanaks)

This repo contains code for out Region-Aware Makeup Transfer using GCNs. 
## Model Architecture:


<img width="2284" height="2640" alt="image" src="https://github.com/user-attachments/assets/1bf8cbde-ebc5-4810-912d-89cb8fc0c5cf" />

<img width="2444" height="2804" alt="gcn" src="https://github.com/user-attachments/assets/33a84759-c63c-437d-836c-91a2651c77f6" />

## Results:
<img width="876" height="435" alt="image" src="https://github.com/user-attachments/assets/9c96b1eb-e36e-4bee-a862-691d70cd6e8b" />

## Evaluation through user review:
Mean score: 4.9/10

Mean score of images by 54 users:
<img width="1200" height="742" alt="image" src="https://github.com/user-attachments/assets/742ed75a-c603-41e8-ba52-d383b145a077" />



### XFlow-GAT
 XFlow-GAT was the first architecture we experimented with. The model followed a multi-stage pipeline:
node features → graph message passing → updated node features → rasterization → U-Net → image.
The idea was to let a GAT capture fine-grained interactions between corresponding facial landmarks and then decode those refined node embeddings into a full image. In practice, this design turned out to be unnecessarily complicated for the relatively constrained task of region-based makeup transfer. The model hallucinated unnatural colors, oversmoothed key regions, and failed to maintain global consistency.
The figure below shows an example of the output produced by XGAT-Color. The overly smooth eyelids, distorted lip coloration, and color bleeding into the background demonstrate the instability of this approach when used for direct makeup transfer.

<img width="1083" height="361" alt="image" src="https://github.com/user-attachments/assets/5ba13c10-308e-4220-bf40-1968c5daae73" />




