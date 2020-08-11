# Interactive-StyleGAN
Applying the DeepSIE algorithm on Stylegan

Code for the Master Thesis Deep Interactive Evolutionary Systems for Assisted Creativity

Problem. Deep Interactive Evolution combines the capacity of Interactive Evolutionary Computation to capture user’s preference with the domain-specific robustness of a trainedGAN  generator,  allowing  the  user  to  control  the  GAN  output  by  evolving  latent  vectors through simulated breeding.  However, there has been no research on the potential of deep interactive evolutionary systems in the realm of AI-assisted artistic creation and creative exploration. In this thesis, we explore that possibility. 

Model solution. We implement DeepIE within a Style-Based generator from a Style-GAN model trained on the WikiArt dataset.  We also propose StyleIE, a variation of DeepIE that takes advantage of the secondary latent space in the Style-Based generator.  Traditional GAN latent spaces must follow the distribution of the training data, which results in an undesired phenomena called “entanglement”, where the features of the output images can not be independently controlled. The Style-Based generator employs a mapping network to learn a “disentangled” intermediate latent space that does not follow the original data distribution. The AdaIN operator is employed to allow vectors in the “disentangled” latent space independent control over specific features or styles in the output images. We adapt DeepIE to work in this secondary latent space and call this new approach Style-Based DeepIE or StyleIE. 

Experiments. Both original DeepIE and StyleIE are tested with volunteers from the animation and design school.  We performed two experiments.  In experiment 1 we had 20volunteers and gave them the goal of generating an expressionist portrait using the systems. In experiment 2 we had 40 volunteers and the goal was to generate a painting resembling “The Scream” by Edvard Munch.  For each experiment, we divided the users in two groups: AB and BA. AB first performed the assigned task using DeepIE, then performed it a second time using StyleIE. The BA group started with StyleIE, then used DeepIE the second time. Self-rated evaluation of the results was collected through a questionnaire and the qualities of the resulting final images were analysed using deep convolutional neural models:  RASTA, NIMA and Deep Ranking.

Results. In experiment 1, user self-rated evaluation and CNN-based analysis showed no difference in the performance between DeepIE and StyleIE. In experiment 2, StyleIE out-performed DeepIE in both self reported and CNN-based metrics.  Our findings suggest thatStyleIE and DeepIE perform equally in tasks with open-ended goals with relaxed constraints, but StyleIE performs better in more close-ended and constrained tasks. We conclude that deep interactive evolutionary systems are promising tools for artistic and creative practice.
