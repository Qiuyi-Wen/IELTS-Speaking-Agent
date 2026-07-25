# Model evaluation set

`cases.json` contains 12 fixed text answers across low, mid, and high quality
levels. Each case defines an expected text-score range and, for deliberately
incorrect answers, issue categories that the specialist judges should detect.

The deterministic test suite validates the dataset and comparison logic without
calling a model:

```bash
python -m unittest discover -s tests -v
```

Live evaluation makes paid DashScope calls and is therefore opt-in:

```powershell
$env:DASHSCOPE_API_KEY = "your-key"
python -m evals.run_model_evals --limit 3
```

Run specific cases:

```bash
python -m evals.run_model_evals --case-id low_place_01 --case-id high_place_01
```

The command exits with status `1` when a score falls outside its expected range
or a required issue category is missed. Start with a small limit because every
case invokes the grammar judge, vocabulary judge, and head coach.
