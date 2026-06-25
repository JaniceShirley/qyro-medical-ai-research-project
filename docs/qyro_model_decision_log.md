# QYRO Model Decision Log

This log documents the design rationale behind selecting the specific model backbones for **QYRO Acne v1** and lists the alternatives that were evaluated and rejected.

---

## 1. Model 1: Lesion Detection (YOLOv8n)

We selected **YOLOv8n** (nano) for the localized object detection task.

### 1.1 Architecture Rationale
* **Parameters**: 3.2M parameters.
* **Latency**: $<10$ms on modern mobile CPU.
* **Compatibility**: Exportable to ONNX, CoreML, and TensorFlow Lite (TFLite) out-of-the-box.
* **Anchor-Free Design**: Eliminates the need for hand-tuned anchor box ratios, resulting in faster and more accurate localization of small comedones and papules.

### 1.2 Rejected Alternatives

| Model | Rationale for Rejection |
| :--- | :--- |
| **Faster R-CNN (ResNet50 Backbone)** | **Too Heavy**: $>40$M parameters. Slower inference (often $>150$ms on edge CPUs), making interactive smartphone scanning laggy and unresponsive. |
| **SSD (Single Shot Detector - MobileNetV2)** | **Poor Localization for Small Lesions**: While lightweight, SSD struggles to detect micro-comedones and tiny papules due to its rigid anchor box grids and lack of fine feature maps. |
| **YOLOv8x (Extra Large)** | **Overkill & Overfitting risk**: 68.2M parameters. The high capacity is unnecessary for our single-class detection dataset (Kurnaz, 520 images) and leads to immediate overfitting. |

---

## 2. Model 2 & 3: Subtype Classification & Severity Grading (EfficientNet-B0)

We selected **EfficientNet-B0** for both the subtype classification and ordinal severity grading models.

### 2.1 Architecture Rationale
* **Parameters**: 5.3M parameters.
* **Compound Scaling**: Scales width, depth, and resolution uniformly using a neural architecture search (NAS) base.
* **Data Efficiency**: EfficientNet-B0 is highly responsive to transfer learning and achieves top accuracy on relatively small datasets compared to standard CNNs.
* **Interpretability**: Lighter depth allows for highly accurate Grad-CAM saliency mapping, which is essential for medical explainability.

### 2.2 Rejected Alternatives

| Model | Rationale for Rejection |
| :--- | :--- |
| **ResNet-50** | **High Parameter Footprint**: 25.6M parameters. ResNet50 converges slower than EfficientNet and has more than 4x the parameter count, which increases mobile app bundle sizes and limits edge deployment. |
| **MobileNetV3-Large** | **Marginal Accuracy Loss**: While highly optimized for mobile (5.4M parameters), MobileNetV3-Large exhibited slightly worse representation learning compared to EfficientNet-B0 on our fine-grained derm datasets. |
| **Vision Transformers (ViT-B/16)** | **Data Hungry & High Latency**: ViTs require massive pretraining datasets (like JFT-300M or ImageNet-21k) to generalize because they lack spatial inductive bias. On limited derm datasets, they tend to overfit and are computationally prohibitive to run on CPU edge devices. |
