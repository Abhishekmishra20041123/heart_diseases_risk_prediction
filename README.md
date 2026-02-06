# Heart Disease Detection Dashboard

This repository contains a comprehensive Streamlit dashboard for heart disease detection using multiple machine learning models. The dashboard provides interactive visualizations, model performance comparison, and real-time predictions.

## 🚀 Features

- **Dataset Overview**: Statistical summaries and target distribution analysis
- **Feature Distributions**: Interactive histograms, box plots, and violin plots
- **Correlation Analysis**: Interactive heatmaps showing feature relationships
- **Model Performance Comparison**: Detailed metrics for multiple ML models
- **Real-time Prediction Interface**: Instant risk assessment with probability scores
- **Feature Importance Visualization**: Understanding which factors contribute most to predictions

## 📊 Models Included

- **Logistic Regression**: Classical linear classifier
- **Random Forest**: Ensemble tree-based method
- **XGBoost**: Gradient boosting algorithm
- **CatBoost**: Gradient boosting with categorical feature handling

## 🛠️ Installation

1. Clone the repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## 📋 Usage

The dashboard provides five main views:

1. **Dataset Overview**: Shows dataset statistics and target distribution
2. **Feature Distributions**: Visualizes individual feature distributions by target class
3. **Correlation Analysis**: Shows feature correlations with interactive heatmaps
4. **Model Performance**: Compares all models using accuracy, precision, recall, and F1-score
5. **Prediction**: Real-time prediction interface with risk assessment

## 📁 Project Structure

```
heart_disease_detection/
├── dashboard.py          # Main Streamlit application
├── requirements.txt      # Python dependencies
├── dataest/              # Dataset folder
│   └── heart.csv         # Heart disease dataset
├── models/               # Trained model files
├── README.md            # Project documentation
└── start_dashboard.bat  # Windows startup script
```

## 🎯 Model Performance

Based on proper evaluation with duplicate removal:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CatBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 0.9836 | 0.9706 | 1.0000 | 0.9851 |
| Logistic Regression | 0.6230 | 0.6042 | 0.8788 | 0.7160 |

## 💡 Key Insights

- The dataset contains 302 unique samples after removing 723 duplicate entries
- Ensemble methods (Random Forest, XGBoost, CatBoost) show excellent performance
- Feature importance analysis helps understand critical risk factors
- Real-time prediction provides immediate risk assessment

## 🎥 Demo

A screen recording demonstrating the dashboard functionality is available in the repository as `Screen Recording 2026-02-07 032005.mp4`. The video may not play directly in all browsers due to GitHub's display policies for larger files. Download the file directly from the repository to view the demonstration.

## 🔧 Troubleshooting

- Ensure all model files exist in the `models/` directory
- Check that `heart.csv` is present in the `dataest/` directory
- Install all dependencies from `requirements.txt`
- For any sklearn compatibility issues, ensure you have compatible versions

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

## 📄 License

This project is open source and available under the MIT License.