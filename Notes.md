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
 - [ ] Try and see whether we can actually scrape the article bodies of the FakeNewsNet dataset
 - [ ] Look for papers that perform classification using 2 streams of data
 - [ ] Find a baseline model
