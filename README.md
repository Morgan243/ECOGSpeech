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
```bash
python experiments.py --dropout=0.5 --dropout-2d --n-cnn-filters=32 --n-epochs=35
```
- Add `--results-dir` option to save results in a `./results/` directory
```bash
python experiments.py --result-dir=./results/ --dropout=0.5 --dropout-2d --n-cnn-filters=32 --n-epochs=35
```

#### result_parsing.py
- Load results and possibly model, produce plots to disk
- If --eval-sets is passed, then that set is loaded and the model is run on it to produce predictions
- Without --eval-sets, just results parsable from the JSON file are utilized 
```bash
python result_parsing.py --result-file results_per_patient/1617896854_78dea683-86a4-46c1-b592-5d7b248a0841_TL.json --eval-sets=test
```

