## Analysis of baseball pitcher salaries

### Idea
OpenML has a dataset of a bunch of attributes about baseball pitchers
in 1987. We will create a PSL model that uses standard linear or random forest
regression on that data to predict salary based on attributes. We will then
use PSL to apply some collective predicates, such as having similar names, being
on the same team, etc. Hopefully, our relational model is more accurate than
the standard IID model.

### Installation
You need python3.
```
pip install -r requirements.txt
```

Download dataset:
```
./bin/fetch_data.sh
```

### Usage
Preprocess data into a form PSL accepts as input:
```
./bin/preprocess.sh
```

Run PSL:
```
./bin/run.sh
```

### Notes
