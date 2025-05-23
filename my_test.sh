#!/bin/bash
set -x
partition='INTERN2'
TYPE='reserved' 
# TYPE='spot'  # 可以挤掉
JOB_NAME='yx_internvl'
GPUS=1 
GPUS_PER_NODE=1 
CPUS_PER_TASK=12
export MASTER_PORT=29501
SRUN_ARGS="-w SH-IDC1-10-140-37-153"
CONFIG="./configs_dota15/mcl/mcl_point2rboxv2_dota15_20p.py"
EXTRA_ARGS="--auto-resume"
http_proxy=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.50:23128/
https_proxy=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.50:23128/
HTTP_PROXY=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.50:23128/
HTTPS_PROXY=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.50:23128/
WANDB_API_KEY="b1e5fa15e4cbf702cfeb8ff3bd58a522c908cebc"
# export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
WORK_DIR='./work_dir/dota15/debug/'
CHECKPOINT='./work_dir/dota15/latest.pth'
srun -p $partition --job-name=${JOB_NAME} ${SRUN_ARGS} \
  --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=${CPUS_PER_TASK} -n${GPUS} \
  --quotatype=${TYPE} --kill-on-bad-exit=1 \
  python test.py ${CONFIG} ${CHECKPOINT} --work-dir ${WORK_DIR} --eval mAP --launcher 'none'
