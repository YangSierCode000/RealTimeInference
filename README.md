# TODO

```bash
# phase=train
phase=eval

config=default
# config=res16unet34c

python $phase.py --config=config/scannet/$phase\_$config.gin
```