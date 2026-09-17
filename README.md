# Differentially Private Machine Learning & Cybersecurity Repository

This repository represents the source code, analysis and  results for 12 core projects focused on Differential Privacy and Privacy Preserving Machine Learning. The target is to ensure the guarantee of mathematical privacy bounds (Epsilon) maintaining model utility, text /image classification, fine tuning large language models, using LoRA and projects to protect cyber hacking.

---

##  Tech Stack

* *Frameworks & Libraries:* PyTorch, Opacus, Hugging Face (Transformers, PEFT)
* *Core Concepts:* Differential Privacy (DP-SGD), Privacy-Utility Trade-off, Membership Inference Attack (MIA), LoRA Fine-tuning, Privacy Auditing & Benchmarking
* *Data & Tasks:* Medical Dataset, LLM, NLP (Text Embedding), Computer Vision (MNIST), Synthetic Data Generation

---

##  Projects Overview & Empirical Results

**1. DP-SGD Privacy-Utility Trade-off Analysis**


File Name: dp_sgd_privacy_utility_tradeoff_analysis.ipynb

Objective: Analyze how changes in the noise scale results in the privacy loss (Epsilon) and the model's loss (Utility) .

Results & Achievements: Successfully maintain the mathematical trade-off by varying the Noise Multiplier from 0.5 to 3.0.

Note:  Graphs and trade-off  curves are included inside this notebook.


**2. DP-CNN for MNIST Image Classification**


File Name: dp_cnn_mnist_image_classification.ipynb

Objective: Make a privacy-preserving Convolutional Neural Network (CNN)  on 60,000 real-world images from the MNIST dataset.

Results & Achievements: Resulted a strong and tight privacy guarantee of Epsilon = 0.11, which reduce data leakage risks to near zero.

Note: Training loss curves and privacy-utility  plots are  inside this notebook.



**3. Differentially Private Text Classification**
 

File Name: dp-text-classification-opacus.ipynb

Objective: Make a privacy-preserving text classifier using NLP and Opacus to protect user emails and data.

Results & Achievements: Got a privacy guarantee with a strong upper bound of Epsilon = 1.99.

Note:  Epsilon growth  charts and accuracy plots are inside this notebook.

**4. Membership Inference Attack (MIA) Defense**

File Name: mia-defense-analysis-opacus.ipynb
  
Objective: Protect machine learning models against adversarial threats where an attacker attempts to determine whether a specific data point was used in the training set.
  
Results & Achievements: Integrated a Noise Multiplier of 1.5 using Opacus to successfully neutralize the attack, drastically degrading the hacker's attack success rate to just 51.13% (which is equivalent to random guessing).


**5.DP LLM Fine-tuning using LoRA**

File Name: dp_llm_finetuning_lora_opacus.ipynb

Objective: Protecting medical information in Large Language Model (LLM) . Using LoRA  for parameter tuning and Opacus for differential privacy .

Results & Achievements: Got an exceptional privacy guarantee of Epsilon = 0.5 while fine-tuning the model on a sensitive, original medical dataset.

Note: Fine-tuning parameter plots and mathematical budget tracking charts are inside this notebook.


**6. DP Synthetic Data Generation**

File Name: dp-synthetic-data-generation-mst.ipynb

Objective: Generate private, high quality synthetic datasets conforming to differential privacy definitions.

Results & Achievements: Produced high-utility generative synthetic data that completely shields the underlying original records, rendering it safe for external sharing and research distribution.

Note: Data distribution comparison graphs (Original vs. Synthetic) are  inside this notebook.


**7. Differential Privacy Auditing and Benchmarking: Baseline vs. Low-Epsilon Safety**

File Name: dp_auditing_benchmarking_baseline_vs_safety.ipynb

Objective: Audit and benchmark the privacy leakage of models with a standard non-private baseline against a strong secure, low-epsilon figure using Opacus.

Results & Achievements: Get a  strict privacy bounds of the low-epsilon system. By maintaining a high Noise Multiplier of 3.5, the system hold a secure  privacy bound of Epsilon = 0.3639, with  a near-zero verifiable information  compared to the  privacy leakage of the unprotected baseline model.

Note:  Bar charts mapping Baseline Leakage vs. Low-Epsilon Safety limits are  inside this notebook.


**8. Differential Privacy Auditing via ROC/AUC Curves**
   

File Name: dp_empirical_privacy_auditing_roc_curve.ipynb

Objective: Test and measure how private data in a machine learning model leaks under Membership Inference Attack using ROC curves and  AUC scores.

Results & Achievements: Got a tested  defense system in a real-world setup. Without protection, the baseline model leaked a lot of data with a high AUC of 0.9760. After adding Differential Privacy, the attack failed completely and the rate dropped to an AUC of 0.5374, which is equal to random guessing.

Note: Complete ROC/AUC trade-off plots, false-positive benchmarking matrices, and leakage visualization curves are included inside this notebook.


**09. DP-SGD -Adaptive-Noise-Scheduling**

File name-adaptive_noise_scheduling_dp_sgd.ipynb

Objective: To monitor the training step of the model through 20 epochs by tracking the reduction of the noise multiplier scale and  its impact on the Binary Cross-Entropy Loss for model convergence.

Result & Achievement: The Noise Multiplier Scale decreases following an exponential decay curve, dropping from an initial maximum value of 2.0 at epoch 1 to approximately 1.03 by epoch 20.  The Binary Cross-Entropy Loss goes a sharp decline in the first 3 epochs, falling rapidly from over 0.725 to 0.698, and stands between 0.691 and 0.695 from epoch 5 , giving successful model convergence.

Note: Adaptive Curves are inside the notebook


**10.DP-SGD-Sharding-Machine-Unlearning**

File Name-dp_sharding_machine_unlearning.ipynb

Objective: To compare the execution time (velocity) of the Optimized DP-Sharding Unlearn  against the Traditional Full Retraining method for machine unlearning.

Result & Analysis : The Optimized DP-Sharding Unlearn method  dropped the execution time to just 0.0058 seconds to 0.0184 seconds  by Traditional Full Retraining. This indicates a faster speedup (approximately 3.17x faster), proving the target of the approach in removing data from the trained model.

Note: Adaptive Bar Charts are inside the notebook


**11.DP-SGD-Benchmarking-Pytorch vs jax**


File Name-dp_sgd_speed_benchmarking_pytorch_vs_jax.ipynb

Objective: To  compare the computation velocity (execution time) for 1000 samples under DP-SGD gradient calculation using a traditional PyTorch (Manual Gradient Loop) vs. an optimized JAX (vmap + JIT Compiler) setup.

Result: The JAX (vmap + JIT Compiler)  speeds up computation, taking only 0.0006s, compared to 2.4649s  by the PyTorch (Manual Gradient Loop). This represents a huge acceleration (approximately 4108x faster) giving the  efficiency of JAX's vectorized and just-in-time compilation capabilities for per-sample gradient clipping in DP-SGD .

Note : Adaptive Curves are inside the notebook


**12.DP-Federated-Learning-non-iid-Data**

File Name-dp_federated_learning_non_iid_data.ipynb

Objective: To compare the training loss of two client groups over 5 rounds and To evaluate model convergence under private and non-IID data conditions.

Result: Non-IID data and privacy noise cause high fluctuations in both loss curves and Client 1 loss increases at round 5, while Client 2 successfully reaches its minimum loss.

Note: Curves are inside the notebook








**Research Manuscripts are in Preparation**


**1. Accelerating Per-Sample Gradient Clipping in DP-SGD: A High-Performance Benchmarking Approach
using JAX and Adaptive Noise Scheduling**

Abstract— It is important to use DP-SGD in Differential Privacy to ensure the privacy of data, but it reduces the speed of model training. The main cause of reducing the speed is calculating each sample's gradient and counting per-sample gradient clipping. In frameworks like PyTorch, to perform this work, it is essential to apply a manual loop in the backend. To solve this problem, I use a high-optimization pipeline. Here, I use vmap in JAX and a just-in-time (JIT) compiler. Again, I use Exponential Decay-based Adaptive Noise Scheduling in 20 epochs. The result I get is that this JAX method decreases the time from 2.4649 seconds (PyTorch Baseline) to 0.0006 seconds. It increases the speed of computation to 4108x. Besides this, it also decreases the Binary Cross-Entropy loss from 0.725 to 0.693 and finally converges the model. At last, by using JAX, we can remove the computational loss and slowness of the model.

**2. Empirical Privacy Auditing of Deep Learning Models: Benchmarking Membership Inference Attacks
and Defenses via Opacus**

Abstract— The main problem of deep learning models is the Membership Inference Attack (MIA). By making this attack, a hacker can easily find out if a specific data point was used in the training set or not. If we train the baseline model without any protection, it leaks data, which results in a rise in the AUC score to 0.9760, meaning the hacker's success rate is near 100%. To audit and defend this privacy leakage, I use PyTorch Opacus to generate a less epsilon-based security system and auditing framework. I set a noise multiplier from 1.5 to 3.5 to check the model's defense. As a result, I get a strict privacy bound which is Epsilon = 0.3639. This destroys the attacking power of hackers. By adding defense, the score of the AUC score decreases from 0.9760 to 0.5374 and the hacker's success rate becomes 51.13% which means the hacker's attack is equivalent to random guessing. At last, it is proved that the proper use of differential privacy auditing and Opacus can reduce the percentage of information leakage to zero.

**3. Privacy-Preserving Fine-Tuning and Data Synthesis: Scaling Differential Privacy to Complex Modalities and Sensitive Medical Data**

Abstract— In the medical domain, when we use large language models, there is a problem that sensitive medical information can leak if a hacker attacks the system. For the medical privacy law, it is totally prohibited to share medical data records outside. To solve this problem, I use a Dual-Layer Privacy method. First, I use LoRA (Low-Rank Adaptation) and PyTorch Opacus on a sensitive medical dataset to fine-tune the LLM model. Second, to secure data sharing outside, I use the MST (Maximum Spanning Tree) method and generate differential privacy private high-quality synthetic data. As a result, when I combine the tuning method using LoRA and Opacus, I get a unique and strong privacy guarantee of Epsilon = 0.5. My generated synthetic data highly matches the real data distribution (high utility) as well as it hides the original data. At last, it can be said that if we use fine-tuning and synthetic data generation together, it will secure internal and external research in medical dataset information.

**4. Scalable Data Governance in Trustworthy Machine Learning: Sharding for Accelerated Machine Unlearning and Non-IID Federated Convergence**

Abstract— In the present day, to ensure the "Right to be Forgotten," it becomes essential to perform Machine Unlearning. But traditionally, if we want to remove data from a model, we have to do full retraining, which is very slow and expensive. On the other side, in a Federated Learning system, when the clients' data are different, meaning that they are Non-IID Data, then it is difficult to converge the model training and reduce loss due to privacy noise. To estimate these two challenges, in this paper, I propose a scalable data governance framework. First, I use an Optimized DP-Sharding Unlearn method which ensures data unlearning without training the whole model. Second, I track the training loss convergence throughout 5 communication rounds under privacy noise and Non-IID conditions. My experimental results show that the execution time of the Optimized DP-Sharding method reduces from 0.0184 seconds to 0.0058 seconds, which means that it increases the speed 3.17x faster. In reality, we can see huge fluctuations in the loss curve, but in my system, Client 2 reaches its minimum loss successfully. At last, it is proved that a sharding-based method and right convergence tracking make the data governance and privacy work of trustworthy machine learning systems more scalable and realistic.





---

## 💻 How to Run

1. Clone the repository and navigate to the project directory:
git clone https://github.com
cd privacy-preserving-deep-learning

2. Install all required dependencies:
pip install torch torchvision opacus notebook matplotlib numpy transformers peft

3. Launch Jupyter Notebook or upload the files to Google Colab to run any of the project scripts:
jupyter notebook

---

## 📄 License
This repository is licensed under the [MIT](LICENSE) License.
