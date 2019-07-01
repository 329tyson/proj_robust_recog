# proj_robust_recognition

## Training how-to

Training default settings are in *config.json*
Once you clone the repository, you just need to change
- *imagepath*
- *train_label*
- *test_label*
in *config.json*

then export config.json path to environment variable
```sh
export CONFIG_PATH=path_to_config.json
```

and run (inside model/ directory)
```sh
python trainer.py
```
