# 🚀 Quick Start Guide

## Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd TRI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Integrated System
```bash
python final_dash.py
```

## 🎯 Key Features

### Lane Detection
- **Hybrid Approach**: Combines traditional CV with deep learning
- **Success Rate**: 95.5% lane detection accuracy
- **Real-time**: 2.4 FPS processing speed

### Object Detection & Tracking
- **YOLOv5**: State-of-the-art object detection
- **DeepSORT**: Multi-object tracking
- **Vehicle Focus**: Cars, trucks, buses, motorcycles

### Bird's Eye View
- **Perspective Transform**: Real-time BEV mapping
- **Object Projection**: 3D to 2D coordinate transformation
- **Visualization**: Top-down view with object positions

## 🖥️ System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8+
- **RAM**: 4GB+ recommended
- **GPU**: CUDA-compatible (optional)
- **Storage**: 2GB+ free space

## 🎮 Controls

- **'q'**: Quit application
- **'d'**: Toggle debug panel
- **Mouse**: Calibration mode (first run)

## 📊 Performance Metrics

| Component | Success Rate | Processing Time |
|-----------|-------------|----------------|
| Lane Detection | 95.5% | 0.274s |
| Object Detection | >90% | 0.080s |
| Tracking | >90% | 0.056s |
| **Total System** | **95.5%** | **0.409s** |

## 🔧 Troubleshooting

### Common Issues
1. **CUDA Error**: Install CPU-only PyTorch
2. **Model Loading**: Check file paths in code
3. **Performance**: Reduce video resolution
4. **Memory**: Close other applications

### Debug Mode
Enable debug panel to see:
- Road segmentation mask
- Edge detection results
- Sliding window search
- Lane fitting process

## 📈 Next Steps

1. **Calibrate System**: Run calibration for your camera
2. **Test Performance**: Monitor FPS and success rates
3. **Customize Parameters**: Adjust thresholds for your use case
4. **Extend Functionality**: Add new features or models

## 📚 Documentation

- [Full Documentation](README.md)
- [Project Summary](PROJECT_SUMMARY.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Technical Details](setup.py)

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Review documentation
3. Open an issue on GitHub
4. Contact: pranav1gilda@gmail.com
