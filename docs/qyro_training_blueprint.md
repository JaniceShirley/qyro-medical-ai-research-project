# QYRO Acne AI Training Blueprint

This document specifies the training protocols, loss functions, optimization strategies, and validation setups for the three models in the **QYRO Acne v1** pipeline.

---

## 1. Model 1: Lesion Detection (YOLOv8n)

### 1.1 Preprocessing & Augmentations
* **Input Size**: $640 \times 640$ pixels, padded via letterbox to preserve aspect ratio.
* **Normalizations**: Pixels scaled to $[0.0, 1.0]$.
* **Augmentation Pipeline (Albumentations)**:
  * Horizontal flip (probability = 0.5)
  * **Vertical flip is strictly disabled** (vertical flipping changes clinical/spatial contextual layout).
  * Mosaic augmentation (disabled during the last 10 training epochs).
  * Random HSV adjustments (fractional variations in saturation, value, and hue).
  * Scale and translation transforms.

### 1.2 Loss Formulation
The YOLOv8 loss is composed of three components:
1. **Bounding Box Regression Loss (CIoU)**:
   $$\mathcal{L}_{\text{box}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha \cdot v$$
   where $c$ is the diagonal distance of the smallest enclosing box, $\rho^2$ is the euclidean distance between bounding box centers, and $v$ measures aspect ratio consistency.
2. **Distribution Focal Loss (DFL)**:
   Penalizes bounding box borders that deviate from the target distributions:
   $$\mathcal{L}_{\text{dfl}}(S_i, S_{i+1}) = - ((y_{i+1} - y) \log(S_i) + (y - y_i) \log(S_{i+1}))$$
3. **Binary Cross-Entropy (BCE) Loss**:
   Used for class-probability assignment of candidate anchors.

### 1.3 Hyperparameters & Training Config
* **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay = $0.0005$).
* **Base Learning Rate**: $\eta_0 = 0.01$.
* **LR Scheduler**: Cosine decay down to $0.01 \times \eta_0$.
* **Epochs**: 100 epochs.
* **Batch Size**: 32 samples.
* **Warmup Epochs**: 3 epochs (warmup bias LR = 0.1).

---

## 2. Model 2: Subtype Classification (EfficientNet-B0)

### 2.1 Preprocessing & Augmentations
* **Input Size**: $224 \times 224$ pixels.
* **Augmentations**:
  * Horizontal flip (probability = 0.5)
  * **Vertical flip is strictly disabled**.
  * Albumentations `CoarseDropout` (simulating localized skin occlusions or hair shadows).
  * Random brightness and contrast adjustment ($p=0.4$).
  * Grid Distortion ($p=0.2$).

### 2.2 Class Imbalance Mitigation
Due to high class imbalance in the subtype classification dataset (e.g., `mechanica` and `scar` have very few samples), we compute class-weights to scale the loss:
$$w_c = \frac{N}{C \cdot N_c}$$
where $N$ is total images, $C$ is the number of classes (9), and $N_c$ is the image count of class $c$.

### 2.3 Loss Function: Class-Weighted Cross-Entropy
$$\mathcal{L}_{\text{subtype}} = - \frac{1}{M} \sum_{j=1}^{M} \sum_{c=1}^{C} w_c \cdot y_{jc} \log(p_{jc})$$
where $y_{jc}$ is the binary indicator for sample $j$, class $c$, and $p_{jc}$ is the predicted probability.

### 2.4 Hyperparameters & Transfer Learning Strategy
* **Pretrained Weights**: ImageNet-1k initialization.
* **Phase 1 (Feature Extraction)**: Freeze the backbone features, train only the linear classification head for 10 epochs. Learning rate $\eta_{\text{head}} = 0.001$.
* **Phase 2 (Fine-Tuning)**: Unfreeze all layers, train for 40 epochs. Learning rate $\eta_{\text{ft}} = 0.0001$ with cosine decay.
* **Batch Size**: 64.

---

## 3. Model 3: Severity Grading (EfficientNet-B0 Ordinal Regression)

### 3.1 Preprocessing & Augmentations
* **Input Size**: $224 \times 224$ pixels.
* **Augmentations**: Identical to Model 2 (no vertical flips).

### 3.2 Ordinal Loss Formulation
We use ordinal regression to penalize larger class prediction gaps heavier than closer ones. For $K=4$ severity stages, the network predicts $K-1 = 3$ binary logits.
For a target severity stage $y \in \{1, 2, 3, 4\}$, the binary target vector $\mathbf{t} = [t_1, t_2, t_3]^T$ is populated:
$$t_i = \begin{cases} 1 & \text{if } y > i \\ 0 & \text{otherwise} \end{cases}$$
The loss function is the average of binary cross-entropy losses over the ordinal tasks:
$$\mathcal{L}_{\text{severity}} = - \frac{1}{K-1} \sum_{i=1}^{K-1} \left[ t_i \log(\sigma(x_i)) + (1 - t_i) \log(1 - \sigma(x_i)) \right]$$
where $x_i$ is the raw output logit of the network for node $i$, and $\sigma(\cdot)$ is the sigmoid function.

### 3.3 Hyperparameters & Training Config
* **Optimizer**: AdamW (weight decay = $0.0001$).
* **Learning Rate**: $\eta = 0.0005$ with Cosine Annealing.
* **Epochs**: 50 epochs.
* **Batch Size**: 64.

---

## 4. Verification and Validation Setup

To ensure robust clinical validation and prevent over-fitting:

### 4.1 Cross-Validation Strategy
* **Model 1 (YOLOv8n)**: Evaluated on the static `valid` and `test` splits defined in the registry.
* **Model 2 & 3 (EfficientNet-B0)**: Stratified 5-Fold Cross-Validation conducted on the pooled training set to stabilize performance estimates across imbalanced subclasses.

### 4.2 Out-of-Distribution (OOD) Robustness Test
* **Evaluation Set**: The `robustness_holdout` split (Google SCIN subset, 205 images) is reserved **strictly for evaluation**. Under no circumstances will this split be used in training or hyperparameter tuning.
* **Evaluation Pipeline**:
  1. Evaluate model accuracy, precision, and recall on the baseline OOD images.
  2. Compute slice-based metrics stratified by Fitzpatrick Skin Type (FST) and Monk Skin Tone (MST).
  3. Synthesize perturbations (Gaussian blur, compression noise) on the OOD images and measure accuracy decay.
