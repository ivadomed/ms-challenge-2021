# `modeling`

This folder includes custom modeling efforts, including training, validation, and test phase. 
Currently, we include implementations for (i) Modified3DUNet, and (ii) TransUNet3D. 
Check [#44](https://github.com/ivadomed/ms-challenge-2021/pull/44) to learn more. The implemented
features can perhaps be best understood by the command arguments as seen at the top of `main.py`.
Please read the `help` section of these arguments to understand the script in its entirety.
Some of these are described in more detail in [#44](https://github.com/ivadomed/ms-challenge-2021/pull/44)
and its sub-PRs [#47](https://github.com/ivadomed/ms-challenge-2021/pull/47) and 
[#48](https://github.com/ivadomed/ms-challenge-2021/pull/48).


## How to Run

We have a lot of command arguments but below call only mentions the most relevant ones and shows
how you can run single-GPU training.

```
export CUDA_VISIBLE_DEVICES=<GPU_ID>
python main.py -dr ~/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed/ -loc -1 -bs <BATCH_SIZE> -nw <NUM_WORKERS> -id <MODEL_ID> -m <MODEL_TYPE> -ne <NUM_EPOCHS> -sl <SEGMENTATION_LOSS> -bal <BALANCING_STRATEGY> -lr <LEARNING_RATE> -t <TASK> -svs <SUBVOLUME_SIZE> -srs <STRIDE_SIZE>
```

We usually want to use the maximum batch-size possible. With `-svs=64` and `-srs=32`, we can use
`-bs=300` on a single NVIDIA RTX A600 GPU. A `-nw=20` also helped speed-up the training. All finished
current experiments were run with `-m=unet`, `-sl=dice`, `-bal=naive_duplication`, and `t=2` but
we aim to experiment more on these in the near-future. We have some new experiments running on
different values for `-svs` and `-srs`.

The script saves the model under the directory specified with `-s` which is `saved_models` by default,
and the name of the `.pt` is given with the following line of code:
```
model_id = '%s-t=%s-gt=%s-bs=%d-sl=%s-bal=%s-lr=%s-svs=%d-srs=%d.pt' % \
            (args.model_id, args.task, args.gt_type, args.batch_size,
             args.seg_loss, args.balance_strategy, str(args.learning_rate),
             args.subvolume_size, args.stride_size)
```

The `model_id` is composed of the most distinguishing command arguments.

To run evaluation, which consists of (i) computing loss metrics (e.g. SOFT and HARD Dice) on the
validation set and (ii) computing ANIMA metrics on the test set (i.e. **test phase**), it suffices
to add `-e` to example command call shown above. However, don't forget that this will skip the
training. Also, only adding `-e` will assume that the desired model is located under `-s` which
is `saved_models` by default. If you have the saved model file elsewhere, use the `-clp` argument
and specify the path of the saved `.pt` file.

Even though the above example command call ran fairly quick, you can experiment with different 
training options as listed below if you desire:
*   **Model-parallel** training can be started respectively with the following command:
    ```
    export CUDA_VISIBLE_DEVICES=<[GPU_IDs]>
    python train.py -id <MODEL_ID> -dr ~/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed/ -loc 0 -nw 0
    ```
*   **Data-parallel** training can be started respectively with the following command:
    ```
    export CUDA_VISIBLE_DEVICES=<[GPU_IDs]>
    python train.py -id <MODEL_ID> -dr ~/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed/ -loc -1
    ```