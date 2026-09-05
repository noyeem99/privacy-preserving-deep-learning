# Differentially Private Machine Learning & Cybersecurity Repository

This repository represents the source code, analysis and  results for 12 core projects focused on Differential Privacy and Privacy Preserving Machine Learning. The target is to ensure the guarantee of mathematical privacy bounds (Epsilon) maintaining model utility, text /image classification, fine tuning large language models, using LoRA and projects to protect cyber hacking.

---

## 🛠️ Tech Stack

* *Frameworks & Libraries:* PyTorch, Opacus, Hugging Face (Transformers, PEFT)
* *Core Concepts:* Differential Privacy (DP-SGD), Privacy-Utility Trade-off, Membership Inference Attack (MIA), LoRA Fine-tuning, Privacy Auditing & Benchmarking
* *Data & Tasks:* Medical Dataset, LLM, NLP (Text Embedding), Computer Vision (MNIST), Synthetic Data Generation

---

## 🚀 Projects Overview & Empirical Results

**1. DP-SGD Privacy-Utility Trade-off Analysis**


File Name: dp_sgd_privacy_utility_tradeoff_analysis.ipynb

Objective: Analyze how changes in the noise scale results the privacy loss (Epsilon) and the model's loss (Utility) .

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

Results & Achievements: Get a privacy guarantee with a strong upper bound of Epsilon = 1.99.

Note:  Epsilon growth  charts and accuracy plots are inside this notebook.

**4. Membership Inference Attack (MIA) Defense**

File Name: mia-defense-analysis-opacus.ipynb
  
Objective: Protect machine learning models against adversarial threats where an attacker attempts to determine whether a specific data point was used in the training set.
  
Results & Achievements: Integrated a Noise Multiplier of 1.5 using Opacus to successfully neutralize the attack, drastically degrading the hacker's attack success rate to just 51.13% (which is equivalent to random guessing).

**5.DP LLM Fine-tuning using LoRA**

File Name: dp_llm_finetuning_lora_opacus.ipynb

Objective: Hold a medical information in Large Language Model (LLM) . Using LoRA  for parameter tuning and Opacus for differential privacy .

Results & Achievements: Got an exceptional privacy guarantee of Epsilon = 0.5 while fine-tuning the model on a sensitive, original medical dataset.

Note: Fine-tuning parameter plots and mathematical budget tracking charts are inside this notebook.


**6. DP Synthetic Data Generation**

File Name: dp-synthetic-data-generation-mst.ipynb

Objective: Generate private, high synthetic datasets conforming to differential privacy definitions.

Results & Achievements: Produced high-utility generative synthetic data that completely shields the underlying original records, rendering it safe for external sharing and research distribution.

Note: Data distribution comparison graphs (Original vs. Synthetic) are  inside this notebook.


**7. Differential Privacy Auditing and Benchmarking: Baseline vs. Low-Epsilon Safety**

File Name: dp_auditing_benchmarking_baseline_vs_safety.ipynb

Objective: Audit and benchmark the privacy leakage of models with a standard non-private baseline against a strong secure, low-epsilon figure using Opacus.

Results & Achievements: Get a  strict privacy bounds of the low-epsilon system. By maintaining a high Noise Multiplier of 3.5, the system hold a secure  privacy bound of Epsilon = 0.3639, with  a near-zero verifiable information  compared to the  privacy leakage of the unprotected baseline model.

Note:  Bar charts mapping Baseline Leakage vs. Low-Epsilon Safety limits are  inside this notebook.


**8. Differential Privacy Auditing via ROC/AUC Curves**

  File Name: dp_empirical_privacy_auditing_roc_curve.ipynb
  
  Objective: Test and measure how much private data a machine learning model leaks under a Membership Inference Attack by drawing ROC curves and calculating AUC scores.
  
  Results & Achievements: Successfully tested the defense system in a real-world setup. Without protection, the baseline model leaked a lot of data with a high AUC of 0.9760. After adding Differential Privacy,   the attack failed completely and the success rate dropped to an AUC of 0.5374, which is almost equal to random guessing.
  
  Note: Complete ROC/AUC trade-off plots, false-positive benchmarking matrices, and leakage visualization curves are included inside this notebook.

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
