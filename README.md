=# Makeup Transfer Using Graphs

**Members:** [Jiya Sinha](https://github.com/sinhajiya), [Rashi Bharti](https://github.com/rashibharti28), [Sakshi Tiwari](https://github.com/Sakshi-Tiwari2003), [Nandhana KS](https://github.com/nandhanaks)

## HGCN (Histogram guided Makeup Transfer using GCN)



Misc folder: XFlow-GAT

 XFlow-GAT was the first architecture we experimented with. The model followed a multi-stage pipeline:

node features → graph message passing → updated node features → rasterization → U-Net → image.

The idea was to let a GAT capture fine-grained interactions between corresponding facial landmarks and then decode those refined node embeddings into a full image. In practice, this design turned out to be unnecessarily complicated for the relatively constrained task of region-based makeup transfer. The reliance on landmark-level propagation, followed by a rasterization step and a full U-Net decoder, introduced failure modes where the model hallucinated unnatural colors, oversmoothed key regions, and failed to maintain global consistency.

The figure below shows an example of the output produced by XGAT-Color. The overly smooth eyelids, distorted lip coloration, and color bleeding into the background demonstrate the instability of this approach when used for direct makeup transfer.




