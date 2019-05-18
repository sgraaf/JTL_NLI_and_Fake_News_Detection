# Notes
## Meeting 01-05-2019
### Research proposal
We've searched for and examined various datasets that have to do with Stance Prediction and Fake News Detection, and they all come with their caveats (size, nature of the data, etc).
After discussing our findings with the instructor and TAs, we have settled on the following research proposal: **Comparing & contrasting MTL vs Transfer Learning with respect to Fake News Detection (FND) and Natural Language Inference (NLI)**. 
We would do this using the FakeNewsNet dataset (for the FND) and the SNLI dataset (for the NLI)

### Comments about the 2 datasets
Firstly, the FakeNewsNet does not contain the article bodies themselves. They do provide the URLs to those articles & code to perform the scraping needed, but we would have to do so ourselves.
Secondly, the SNLI dataset consists of 572k samples, whereas FakeNewsNet only contains 19k samples (of which 2k are fake and 17k are real). 
It is precisely this class imbalance (in terms of size) that motivates our research proposal.

### Questions
 - Does it make sense to process the article title and body separately (in 2 streams, like with NLI)?
	 - If yes: How do we then classify the article? Do we have to combine the encoded article title and body into a single feature vector? If yes: How??? If no: Then how do we perform classification using these 2 streams?
   - If no: How can we leverage the hidden states and/or outputs produced by the NLI model in our FND model?
 - What would our training strategy be for MTL (NLI first and FND second or the other way around)?
  
### Action points
 - [x] Try and see whether we can actually scrape the article bodies of the FakeNewsNet dataset
 - [x] Look for papers that perform classification using 2 streams of data
 - [ ] Find a baseline model


### Further notes
It might be interesting to consider a model that is described in paper available as `mtl_benchmark.pdf`, where they propose shared layers for the encoder with private classifier layers stacked on top. In this paper, authors incorporated the model to work for **both sentences and pairs of sentences** (however, this does not answer the questions of how to encode a document in our case). The model architecture can be also checked as the image `possible-model.png`.

On the other hand, there is almost no SOTA for the FakeNewsNet that works only with the textual data. I have added some papers with the SOTA results on this dataset (`fakenewsnet_sota.pdf`, `fakenewsnet_social.pdf`, `fakenewsnet_social2.pdf`) but they are not directly comparable/reproducable since they are actually not neural architectures.

Nevertheless, this might only make our research question more legitimate. Furthermore, I was wondering if it is possible to have a 2-stream architecture where first is parsed with the **article text and headline** while the other is parsed with **associated tweets**. Might seem like an interesting research area as well.

Finally, we could consider looking into the LIAR dataset, I have added some papers for reference but don't have any strong thoughts about it yet.


## Meeting 06-05-2019
After further research and discussion, we have now settled on a "hybrid" model, which combines the Multi-Task Deep Neural Network (MT-DNN) set-up a proposed by [Liu et al. (2019)](https://arxiv.org/abs/1901.11504), and the Hierarchical Attention Networks (HAN) document encoder by [Yang et al. (2016)](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf).

We're still not entirely sure on which Dataset to use: While we have been able to succesfully scrape the FakeNewsNet articles using the code provided by the paper authors, SOTA research incorporates the rich social metadata, which is not something we intend to do. We're also exploring the factcheck dataset.




### Action points
- [ ] Implement the sentence encoder based on the HAN architecture
- [ ] Implement the document encoder based on the HAN architecture
- [ ] Implement the SNLI classification based on the practical assignment / other

### Questions
- The HAN architecture uses GRU's for their RNN. Should we also use GRU's or should we use LSTM's? It seems like we could easily swap the one for the other.

### Questions (from May 15)
- How to properly weigh the losses? One approach is to weight them all equally (Augenstein et al., 2018), but this seems not directly applicable for our case.


### Points for future research
- Apply different loss weighting technique (e.g. based on uncertainty, Kendall et al., 2018).




#### References:
Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 7482-7491).
Augenstein, I., Ruder, S., & Søgaard, A. (2018). Multi-task learning of pairwise sequence classification tasks over disparate label spaces. arXiv preprint arXiv:1802.09913.