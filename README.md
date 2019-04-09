# ERFNet-pytorch-chapter5

The chapter5 of the segmentation network summary: 
### Real-time semantic segmentation network.

External links: Efficient ConvNet for Real-time Semantic Segmentation [paper](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf).
ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation [paper](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf).

This chapter is to summarize one of the important directions of semantic segmentation network: Real-time semantic segmentation network. First of all, we summarize the ErfNet mentioned above, which was written in keras framework, but modified in Pytorch framework here. The innovation of this network is to propose a new layer including skip connection and 1D convolutional kernel.

### Environment: 
  
            Pytorch version >> 0.4.1; [Python 2.7]
