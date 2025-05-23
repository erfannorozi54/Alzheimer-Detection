# AD-detection

A comprehensive machine learning pipeline for feature selection and model comparison, designed for Alzheimer's Disease detection. Includes multiple feature selection strategies, model selection, and detailed result logging and visualization.

## Features
- Multiple feature selection methods (Ensemble, RandomForest, Boruta, SI, weighted_SI, multi_resolution_SI, chi2)
- Extensive model comparison (RandomForest, SVM, LogisticRegression, ModelSelector, GradientBoosting, XGBoost, LightGBM, AdaBoost, KNN, NaiveBayes, DecisionTree, Ridge, NeuralNetwork, CatBoost)
- GridSearchCV for hyperparameter tuning
- Detailed logging and result files
- Visualizations for feature selection and model performance

## Project Structure
- `main.py` : Main pipeline script
- `utils/` : Custom modules for feature selection and model selection
- `results/` : Output plots and figures
- `model_selector_inspection.log` : ModelSelector detailed logs
- `feature_selection_comparison_results.csv` : Feature selection results
- `best_feature_selection_methods.csv` : Best feature selection methods summary

## Quickstart (Google Colab)
1. Open `Colab_Run_AD-detection.ipynb` in Google Colab
2. Run all cells to clone the repo, install dependencies, and execute the pipeline

## Manual Run (Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/AD-detection.git
   cd AD-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or, if requirements.txt is missing:
   pip install ucimlrepo scikit-learn xgboost lightgbm catboost matplotlib seaborn pandas
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Outputs
- Results and plots: `results/`
- Logs and CSVs: project root

## License
MIT License

---

**For questions or contributions, please open an issue or pull request.**
