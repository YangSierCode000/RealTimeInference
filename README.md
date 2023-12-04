# Real
This ia an engineering problem, in real world one, not lab thing, need focus on the task.

For a dronez or a robot with constraint ability, the perception range would no larger than a sphere with a diameter `50` meter.

http://semantic-kitti.org/dataset.html#download

https://wandb.ai/yangsiercode000/projects

Low Complexity Tasks or Limited Resources: 1,000 to 20,000 points.
Moderate Complexity: 20,000 to 50,000 points.
High Complexity or High-Resource Availability: 50,000 to 150,000 points.

```bash
conda activate real
cd realtime/

path=nohup
# path=nohup-1
nohup bash train.sh >> $path.out 2>&1 &
```
