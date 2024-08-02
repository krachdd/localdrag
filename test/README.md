# Example of Code for Single Precipitate Case

A virtual python environment is recommended.

The domains given are referd to as *single precipitate test case* in the corresponding publication.

Always use the bash scripts to run the test

```bash
bash run_transform2dto2d.sh 
```

since this includes the linking with `PYTHONPATH` to the localdrag module:

```bash
SOURCEDIR="$(dirname $(dirname "$(pwd)"))"
export PYTHONPATH=$PYTHONPATH:${SOURCEDIR}
```


