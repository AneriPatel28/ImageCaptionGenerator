# Image Caption Generator

This project implements an Image Caption Generator that combines Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) to generate descriptive captions for images. The model is trained on the Flickr8k dataset and utilizes the Xception architecture for feature extraction.

## Features

- **Image Feature Extraction:** Uses the Xception model to extract features from images.
- **Caption Generation:** Employs an RNN to generate captions based on extracted image features.
- **Graphical User Interface:** Provides a user-friendly interface for uploading images and displaying generated captions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/AneriPatel28/ImageCaptionGenerator.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd ImageCaptionGenerator
   ```

3. **Install Dependencies:**

   Ensure you have Python 3.x installed. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Download the Flickr8k Dataset:**

   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k) and place the images in the `Flicker8k_Dataset` directory and the captions file in the `Flickr8k_text` directory.

2. **Preprocess the Data:**

   Run the `preprocess_data.py` script to preprocess the images and captions:

   ```bash
   python preprocess_data.py
   ```

3. **Train the Model:**

   Execute the `train_model.py` script to train the caption generator:

   ```bash
   python train_model.py
   ```

4. **Generate Captions:**

   After training, use the `generate_caption.py` script to generate captions for new images:

   ```bash
   python generate_caption.py --image_path path_to_image
   ```

5. **Launch the GUI:**

   Run the `app.py` script to start the graphical user interface:

   ```bash
   python app.py
   ```

   Upload an image through the GUI to view the generated caption.


## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Flask
- Pillow
- NumPy
- Matplotlib

Install all dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Acknowledgments

This project is inspired by various image captioning research and implementations. Special thanks to the contributors of the Flickr8k dataset.

## License

This project is licensed under the MIT License. 
