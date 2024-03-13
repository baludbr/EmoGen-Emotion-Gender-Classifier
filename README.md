# Deep Learning Audio Classification Project

## Overview

This project aims to classify audio samples into different emotional states using deep learning techniques. The dataset consists of audio recordings labeled with seven different emotions: Anger, Boredom, Disgust, Anxiety, Happiness, Sadness, and Neutral. The classification model is trained on MFCC features extracted from the audio samples.

## Prerequisites

- Python 3
- Libraries:
  - Librosa
  - Pandas
  - Numpy
  - Scikit-learn
  - TensorFlow

## Installation

Clone the repository:

```bash
git clone https://github.com/baludbr/EmoGen-Emotion-Gender-Classifier.git
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation**: Ensure your audio files are organized in the specified structure within the dataset directory.
2. **Feature Extraction**: Run the feature extraction script to extract MFCC features from the audio samples.
   ```bash
   python feature_extraction.py
   ```
3. **Model Training**: Train the classification model using the extracted features.
   ```bash
   python train_model.py
   ```
4. **Evaluation**: Evaluate the trained model on the test set.
   ```bash
   python evaluate_model.py
   ```

## Model Architecture

The classification model architecture consists of convolutional layers followed by fully connected layers. It utilizes the MFCC features as input and outputs probabilities for each emotion class.

## Results

The model achieves an accuracy on the test set.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Librosa](https://librosa.org/doc/main/index.html) - Audio processing library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library

## Contributors

- Balaji Reddy Dwarampudi (@baludbr)
- Revanth Chandragiri (@2100031890)

## Contact

For any inquiries or issues, please contact [dwarampudibalajireddy@gmail.com](mailto:dwarampudibalajireddy@gmail.com).
