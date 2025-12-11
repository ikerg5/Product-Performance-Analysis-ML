# Quick Start Guide

## Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip3 install numpy pandas matplotlib seaborn scikit-learn
```

Or use the requirements file:

```bash
pip3 install -r requirements.txt
```

### Step 2: Run the Analysis

```bash
cd source_code
python3 main.py
```

### Step 3: View Results

Check the `results/` folder for:
- Preprocessing visualizations
- Elbow curve
- Cluster visualizations
- Regression model comparisons
- All statistics CSV files

## What Gets Generated?

### In `data/` folder:
- `cleaned_data.csv` - Preprocessed data with missing values handled
- `normalized_data.csv` - Min-Max normalized features (0-1 scale)

### In `results/` folder:
- `preprocessing_analysis.png` - Data quality visualizations
- `elbow_curve.png` - Optimal k determination
- `cluster_visualization.png` - 2D cluster scatter plot
- `cluster_statistics.csv` - Detailed cluster metrics
- `actual_vs_predicted_linear.png` - Linear regression results
- `actual_vs_predicted_polynomial_2.png` - Polynomial regression results
- `residuals_linear.png` - Residual analysis
- `model_comparison.png` - All models compared
- `model_comparison.csv` - Model performance metrics

## Running Individual Components

### Just Preprocessing:
```bash
cd source_code
python3 preprocessing.py
```

### Just K-means:
```bash
cd source_code
python3 kmeans.py
```

### Just Regression:
```bash
cd source_code
python3 regression.py
```

## Expected Runtime

- Full pipeline: ~30-45 seconds
- Preprocessing: ~5 seconds
- K-means: ~10 seconds
- Regression: ~5 seconds
- Visualization generation: ~10 seconds

## Common Issues

### "ModuleNotFoundError"
**Fix**: Install missing packages
```bash
pip3 install numpy pandas matplotlib seaborn scikit-learn
```

### "FileNotFoundError: results/"
**Fix**: Create the directory
```bash
mkdir results
```

### Running from wrong directory
**Fix**: Make sure you're in `source_code/` when running
```bash
cd source_code
python3 main.py
```

## Understanding the Output

The console will show:
1. **Data Preprocessing**: Missing values, outliers detected, normalization summary
2. **K-means Clustering**: WCSS values for each k, optimal k selection, cluster statistics
3. **Regression Analysis**: Model training progress, performance metrics, comparison

## Key Findings to Report

After running, you'll have:

### Preprocessing Summary:
- Original records: 200
- Final records: 196 (4 dropped due to missing product names)
- Missing values handled
- Outliers detected but kept (premium products)
- Features normalized using Min-Max scaling

### Clustering Results:
- Optimal k will be determined by elbow method (likely 4-5)
- Each cluster will have meaningful names and business insights
- WCSS values showing cluster quality

### Regression Results:
- Comparison of Linear vs Polynomial models
- MSE, MAE, and R² scores for each
- Best model recommendation
- Overfitting analysis

## Next Steps for Your Report

1. **Copy the console output** - It contains all the analysis details
2. **Open the generated images** - Use them in your report
3. **Read the CSV files** - They contain detailed statistics
4. **Interpret the results** - Add your business insights
5. **Document your decisions** - Explain why you chose certain parameters

## Tips for the Assignment

- **Document as you go**: Note why you made preprocessing decisions
- **Understand the visualizations**: Each plot tells a story
- **Business context matters**: Think like a store manager
- **Explain trade-offs**: Why keep outliers? Why this k value?
- **Show comprehension**: Don't just copy outputs, explain what they mean

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Review [assignment_description.pdf](assignment_description.pdf) for requirements
- Use AI tools to understand concepts
- Test incrementally if you make changes

---

**Good luck with your assignment!**
