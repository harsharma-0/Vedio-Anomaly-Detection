# Video Anomaly Detection using Transfer Learning with DenseNet121

This repository contains code for performing video anomaly detection using transfer learning with the DenseNet121 architecture. The goal of this project is to classify anomalies in video frames across multiple classes using deep learning techniques. The code is implemented using TensorFlow and leverages popular libraries for data visualization.

## Table of Contents

1. **Libraries Used**
   - `pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `plotly.express`: Data analysis and visualization
   - `tensorflow`: Deep learning framework
   - `sklearn.preprocessing.LabelBinarizer`, `sklearn.metrics.roc_curve`, `sklearn.metrics.auc`, `sklearn.metrics.roc_auc_score`: Metrics and preprocessing
   - `IPython.display.clear_output`: Clearing output in Jupyter Notebook
   - `warnings`: Managing warnings
   - `os`: Operating system operations
   
2. **Hyperparameters and Directories**
   - `train_dir`: Directory containing training images
   - `test_dir`: Directory containing test images
   - `SEED`: Random seed for reproducibility
   - `IMG_HEIGHT`, `IMG_WIDTH`: Image dimensions
   - `BATCH_SIZE`: Batch size for training
   - `EPOCHS`: Number of training epochs
   - `LR`: Learning rate
   - `NUM_CLASSES`: Number of classes for classification
   - `CLASS_LABELS`: List of class labels
   
## Dataset
To run the code, you will need the video anomaly detection dataset. You can download the dataset from the following Google Drive link: Dataset Download Link.

Place the downloaded dataset in the appropriate directories as specified in the code:

Training data: "C:\\Users\\DELL\\Desktop\\Vdo_Ano_dect\\Train"
Test data: "C:\\Users\\DELL\\Desktop\\Vdo_Ano_dect\\Test"

## Conclusion

This code repository demonstrates the process of video anomaly detection using transfer learning with the DenseNet121 architecture. By following the steps outlined in this README and running the provided code, you can perform training, evaluation, and visualization of the anomaly detection model. Customize the hyperparameters and model architecture to suit your specific dataset and requirements.

For any further assistance or inquiries, please feel free to contact the repository owner.
