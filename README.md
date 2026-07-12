# MNIST-Detect: Single-Digit Object Detection with TensorFlow/Keras

A compact convolutional neural network that **detects and classifies a single handwritten digit** placed at a random location on a blank canvas — combining image classification and bounding-box regression in one end-to-end Keras model.

Built entirely on top of the MNIST dataset (no external annotation tools needed), this project is a hands-on introduction to the core mechanics of object detection: multi-output CNNs, IoU evaluation, and prediction visualization.

---

## 🧠 How It Works

1. **Synthetic dataset generation**
   Each MNIST digit (28×28) is pasted at a random `(x, y)` offset onto a blank 64×64 canvas. The digit's class label and normalized bounding box `[xmin, ymin, xmax, ymax]` are stored as ground truth — turning a plain classification dataset into an object-detection dataset.

2. **Model architecture**
   A 4-block CNN backbone (Conv2D → BatchNorm → MaxPooling, with increasing filter depth: 32 → 64 → 128 → 128) feeds into two output heads:
   - **Classification head** — `softmax` over 10 digit classes
   - **Box regression head** — `sigmoid` output for 4 normalized box coordinates

3. **Multi-task training**
   The model is trained with a combined loss:
   - `sparse_categorical_crossentropy` for digit classification
   - `Huber loss` for bounding-box regression (weighted 5× higher than classification loss)

   `EarlyStopping` and `ReduceLROnPlateau` callbacks are used to stabilize training.

4. **Evaluation**
   - Classification accuracy, per-class precision/recall/F1 (`classification_report`), and a confusion matrix heatmap
   - **Intersection-over-Union (IoU)** statistics for the predicted vs. ground-truth boxes, reported at multiple thresholds (0.50, 0.60, 0.70, 0.75)

5. **Visualization**
   - Sample training images with ground-truth boxes
   - Training curves (accuracy, box MAE, total loss)
   - Side-by-side predicted vs. ground-truth bounding boxes on test images

---

## 📊 Outputs

Running the notebook produces:
- `sample_gt.png` — sample training images with ground-truth boxes
- `confusion_matrix.png` — classification confusion matrix
- `inference_results.png` — predicted vs. ground-truth boxes on test samples

---

## 🏗️ Project Structure

```
.
├── Object_Detection_using_tensorflow.ipynb   # Main notebook
├── sample_gt.png                             # Generated: sample ground-truth visualization
├── confusion_matrix.png                      # Generated: classification confusion matrix
├── inference_results.png                     # Generated: prediction vs ground-truth comparison
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Seaborn

Install dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

---

## 🚀 Usage

1. Clone the repository and open the notebook:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-name>
   jupyter notebook Object_Detection_using_tensorflow.ipynb
   ```
2. Run all cells sequentially. The notebook will:
   - Download MNIST automatically via `keras.datasets.mnist`
   - Generate the synthetic detection dataset
   - Build, compile, and train the model
   - Evaluate classification accuracy and IoU
   - Save visualization plots to the working directory

No GPU is required, though training will be faster with one.

---

## 📈 Results Summary

The model reports:
- **Classification accuracy** on the test set (per-digit precision/recall/F1)
- **Mean IoU** between predicted and ground-truth bounding boxes
- **Detection rate** at IoU thresholds of 0.50, 0.60, 0.70, and 0.75

*(Exact numbers depend on the random seed and training run — see notebook output.)*

---

## 🔍 Key Concepts Demonstrated

- Synthetic dataset creation for object detection from a classification dataset
- Multi-output / multi-task Keras models (`Model` functional API)
- Weighted multi-loss training (classification + regression)
- IoU computation and detection evaluation
- Training callbacks (`EarlyStopping`, `ReduceLROnPlateau`)
- Visualization of predictions vs. ground truth

---

## 📄 License

This project is open-source. Add your preferred license (e.g., MIT) here.

---

## 🙌 Acknowledgments

Built on the classic [MNIST dataset](http://yann.lecun.com/exdb/mnist/), using TensorFlow/Keras.
