# 🖼️ Image Gap Filler - Deep Learning Image Reconstruction Tool

## 🌟 Overview

Image Gap Filler is a Python-based application that uses deep learning techniques to reconstruct and denoise images. The tool provides a graphical user interface (GUI) that allows users to:

- Load images
- Add noise to images
- Train a Convolutional Neural Network (CNN) to reconstruct the original image
- Visualize training progress
- Download the best reconstructed image

## ✨ Features

- **Image Loading**: Select images from your local filesystem
- **Noise Generation**: 
  - Gaussian noise addition
  - Gap-based noise insertion (simulating damaged or incomplete images)
- **Deep Learning Reconstruction**:
  - Configurable CNN architecture
  - Normalized Root Mean Square Error (NRMSE) loss function
  - Real-time training visualization
- **Interactive GUI**:
  - Display original, noisy, and reconstructed images
  - Show training progress and loss curve
  - Download best reconstruction

## 🛠️ Prerequisites

- Python 3.7+
- Libraries:
  - NumPy
  - PyTorch
  - Tkinter
  - Pillow (PIL)
  - Matplotlib
  - torchvision

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Harsh3304/Filler.exe.git
   ```

2. Install required dependencies:
   ```bash
   pip install numpy torch torchvision pillow matplotlib
   ```

## 🚀 Usage

1. Run the application:
   ```bash
   python image_gap_filler.py
   ```

2. Use the GUI buttons:
   - **Add Image**: Select an image to process
   - **Denoise**: Start the CNN training process
   - **Download**: Save the best reconstructed image

## 🖼️ Screenshots

1. **Home Page (for new user)**  
   <img src="/Dependencies/readme_images/results.png" alt="Home Screen Page" width="700">  
## 🔬 Technical Details

- **Image Processing**:
  - Resizes images to 256x256 pixels
  - Normalizes pixel values
  - Adds Gaussian and gap-based noise

- **Neural Network**:
  - Configurable Convolutional Neural Network
  - Adjustable layers, channels, and kernel size
  - Adam optimizer
  - NRMSE loss function for reconstruction quality

## 🛠️ Customization

You can modify the CNN architecture by changing parameters in the `CNN_configurable` class:
- `n_lay`: Number of convolutional layers
- `n_chan`: Number of channels
- `ksize`: Kernel size

## ⚠️ Limitations

- Current version supports RGB images
- Training can be computationally intensive
- Reconstruction quality depends on noise level and image complexity

## 🤝 Contributing

Contributions are welcome! Please submit pull requests or open issues to suggest improvements or report bugs.

## 📄 License

[Specify your license here, e.g., MIT License]

## 🙏 Acknowledgments

- PyTorch Community
- Machine Learning Research Inspiration

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

Harsh Patel - harshp3304@gmail.com

Project Link: [https://github.com/Harsh3304/Filler.exe](https://github.com/Harsh3304/Filler.exe)