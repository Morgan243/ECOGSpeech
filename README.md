## ASPEN + MHRG ECOG Processing Library
See notbooks for example walkthroughs of internal processing and example usage


### Notebooks
#### NW Words Processing Overview
Walkthrough of how Single-word/NorthweesternWords -style data is handled and processed

#### Northwestern Example 
Example of practical development usage of the package, e.g. iterating in the notebook
- Load data
- Build models
- Determine performance

### Command line entry points
Pass `--help` for detailed usage
#### experiments.py
- Run basic configrutaions of Base models and Northwestern words.
- Example usage:
```python
python experiments.py --dropout=0.5 --dropout-2d --n-cnn-filters=32 --n-epochs=35
```
- Add `--results-dir` option to save results in a `./results/` directory
```python
python experiments.py --result-dir=./results/ --dropout=0.5 --dropout-2d --n-cnn-filters=32 --n-epochs=35
```

