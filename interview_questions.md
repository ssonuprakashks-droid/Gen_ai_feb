# Generative AI Professional Program - Interview Questions

A comprehensive collection of interview questions covering foundational to advanced topics in AI, Machine Learning, and Generative AI. This document serves as a preparation guide for technical interviews and assessments.

---

## 1. AI, ML, DL Fundamentals
1. What is the difference between Artificial Intelligence, Machine Learning, and Deep Learning?
2. Why is Deep Learning considered a subset of Machine Learning?
3. Why do traditional ML models fail at generating new data?
4. Explain discriminative vs generative models with examples.
5. Why has Generative AI become practical only in recent years?
6. What role does data scale play in Deep Learning success?
7. Why is Deep Learning considered representation learning?

---

## 2. Generative AI Concepts
8. What does “generative” mean in Generative AI?
9. How does a generative model learn data distribution?
10. Why probabilistic modeling is essential for Generative AI?
11. How is a generative chatbot different from a rule-based chatbot?
12. Why are transformers preferred for generative tasks?
13. What is tokenization and why is it critical for language models?

---

## 3. Mathematics for AI
14. Why is probability fundamental to generative models?
15. Explain the role of mean and variance in neural networks.
16. Why is linear algebra central to Deep Learning?
17. What does a dot product represent in neural networks?
18. Why normalization helps training stability?
19. What happens when gradients vanish or explode?

---

## 4. Neural Networks & Backpropagation
20. How does backpropagation work conceptually?
21. Why do neural networks require non-linear activation functions?
22. Why is ReLU preferred over sigmoid and tanh?
23. What are loss functions and why are they needed?
24. How does gradient descent optimize neural networks?
25. Why deeper networks are harder to train?

---

## 5. Convolutional Neural Networks (CNNs)
26. Why are CNNs better than fully connected networks for images?
27. How does convolution reduce parameters?
28. What is translation invariance?
29. What is the role of pooling layers in CNNs?
30. How does receptive field grow in deep CNNs?

---

## 6. Recurrent Neural Networks (RNNs, LSTMs, GRUs)
31. Why do RNNs struggle with long-term dependencies?
32. What causes vanishing gradients in RNNs?
33. How do LSTM gates solve this problem?
34. What is the difference between LSTM and GRU?
35. When would you prefer GRU over LSTM?

---

# WEEK 2 – Autoencoders & Variational Autoencoders (VAEs)

## 7. Autoencoders
36. What is an autoencoder?
37. Why doesn’t an autoencoder learn identity mapping?
38. What is a bottleneck layer?
39. How does bottleneck size affect reconstruction?
40. Difference between PCA and autoencoders?
41. Why autoencoders are unsupervised models?

---

## 8. Variational Autoencoders (VAEs)
42. Why do we need probabilistic latent spaces?
43. What is latent space representation?
44. Why VAEs generate smoother outputs than GANs?
45. What is KL divergence intuitively?
46. Why is KL divergence added to the loss?
47. Explain reconstruction loss vs KL loss.
48. What is the reparameterization trick?
49. Why can’t we backpropagate through random sampling directly?
50. What happens if KL loss dominates training?

---

## 9. VAE Applications & Comparisons
51. How VAEs are used for anomaly detection?
52. Why VAEs are useful in image generation?
53. Can VAEs be used for data compression?
54. Why VAE outputs are often blurry?
55. When would you prefer VAE over GAN?

---

# WEEK 3 – GANs & Transformers

## 10. Generative Adversarial Networks (GANs)
56. What is the core idea behind GANs?
57. Role of Generator and Discriminator?
58. Why GAN training is unstable?
59. What is mode collapse?
60. Why discriminator should not overpower generator?
61. Difference between GAN and VAE?
62. What is DCGAN and why it is better?
63. How does Conditional GAN work?
64. Why CycleGAN does not require paired data?
65. Ethical concerns with GAN-generated content?

---

## 11. Transformers & Attention Mechanisms
66. Why attention is better than recurrence?
67. What problem does self-attention solve?
68. What is the intuition behind attention mechanism?
69. Why transformers are parallelizable?
70. What is positional encoding?
71. Why transformers scale better with data?
72. Difference between encoder-only, decoder-only, and encoder-decoder architectures?

---

# WEEK 4 – Large Language Models, RAG & Diffusion Models

## 12. Large Language Models (LLMs)
73. What is a Large Language Model?
74. Difference between encoder-based and decoder-based models?
75. Why GPT-style models are decoder-only?
76. What happens during pre-training of LLMs?
77. What changes during fine-tuning?
78. Why instruction tuning improves model responses?
79. What is prompt engineering?

---

## 13. Retrieval-Augmented Generation (RAG)
80. Why do LLMs hallucinate?
81. What is hallucination in LLMs?
82. How does RAG reduce hallucinations?
83. Difference between RAG and fine-tuning?
84. What are embeddings?
85. Why cosine similarity is used?
86. How vector databases work?
87. How would you design a document-based QA system?

---

## 14. Diffusion Models
88. What is diffusion model intuition?
89. Explain forward diffusion process.
90. Explain reverse diffusion process.
91. Why predicting noise works?
92. Why diffusion models are more stable than GANs?
93. Why diffusion models are slow?
94. Tradeoff between inference steps and quality.
95. How Stable Diffusion differs from GAN-based image generation?

---

# PROJECT-BASED & SYSTEM DESIGN QUESTIONS

## 15. Project-Based & System Design Questions

96. Why did you choose this model for your project?
97. How did you preprocess your data?
98. How do you evaluate generative models?
99. Why accuracy is not sufficient for generative tasks?
100. What metrics fail for generative AI?
101. How would you improve model quality?
102. How would you reduce overfitting?
103. What were the major challenges you faced?
104. How would you scale this system?
105. How would you deploy this model?
106. What are the limitations of your project?
107. How would you handle biased outputs?
108. How would you make your system production-ready?
109. How would you reduce inference latency?
110. How would you monitor model performance post-deployment?

---