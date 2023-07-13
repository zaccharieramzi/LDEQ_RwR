# [Recurrence without Recurrence: Stable Video Landmark Detection with DEQs](https://arxiv.org/abs/2304.00600)

This repo includes inference code for our Landmark Deep Equilibrium Model network (LDEQ), which was developed during an internship at **Nvidia** by myself, Pavlo Molchanov, Arash Vahdat, Hongxu Yin and Jan Kautz.

The weights we provide here were reproduced on different hardware than the hardware used for the paper experiments, and results may be slightly different.

## TLDR
This work uses deep equilibrium models to add a form of recurrence at test time, without having access to a recurrent loss at train time. This can be used to improve temporal coherence in video landmark detection when the model is trained on still images.

Please check out this video for a demo:

[![demo](https://img.youtube.com/vi/8Mmpc_-oP6w/0.jpg)](https://www.youtube.com/watch?v=8Mmpc_-oP6w)


## Environment setup

```
conda create -y -n ldeq python=3.8
conda activate ldeq
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python==4.6.0.66 matplotlib==3.5.1 scipy==1.7.3 torchinfo==1.7.0 pandas
```
I haven't tried other versions of these librairies so use them at your own risk.
NB: if `opencv` gives weird errors, `pip uninstall opencv-python` and then `pip install opencv-python-headless==3.5.5.64`


## Test LDEQ on WFLW
1) Download the WFLW dataset that was used for training and testing LDEQ on WFLW. We used the precropped version from the [HIH repo](https://github.com/starhiking/HeatmapInHeatmap) and it can be downloaded there (WFLW.zip).
2) (optional) Check dataset loading is good by running datasets/WFLW/dataset.py, after updating path there
3) Download the LDEQ trained weights [here](https://drive.google.com/drive/folders/1r0NJBXtAW2mIT30bPw83ZgDuPgIUdN9K?usp=share_link). This particular model uses the Anderson acceleration solver.
4) run `python test_LDEQ_WFLW.py --landmark_model_weights /path/to/final.pth.tar --dataset_path /path/to/WFLW --workers 4 --batch_size 32`

### Expected results
The results should be similar to the following:
```
------------ test ------------
NME %: 3.946569064313631
FR_0.1% : 2.400000000000002
AUC_0.1: 0.6212493333333333
Finished loading WFLW dataset

------------ test_largepose ------------
NME %: 6.858957261706837
FR_0.1% : 12.576687116564422
AUC_0.1: 0.3751687116564418
Finished loading WFLW dataset

------------ test_expression ------------
NME %: 4.057816854798313
FR_0.1% : 1.5923566878980888
AUC_0.1: 0.6068694267515924
Finished loading WFLW dataset

------------ test_illumination ------------
NME %: 4.287161939971233
FR_0.1% : 2.005730659025784
AUC_0.1: 0.6288777459407832
Finished loading WFLW dataset

------------ test_makeup ------------
NME %: 3.8494616120731573
FR_0.1% : 0.9708737864077666
AUC_0.1: 0.6201165048543689
Finished loading WFLW dataset

------------ test_occlusion ------------
NME %: 4.78428865761305
FR_0.1% : 5.29891304347826
AUC_0.1: 0.5468401268115942
Finished loading WFLW dataset

------------ test_blur ------------
NME %: 4.594755396480498
FR_0.1% : 2.587322121604141
AUC_0.1: 0.5707559292798621
Total time: 1h06m28s
```


## Test LDEQ on WFLW-V

### WFLW-V download

### Option 1 (Easy)
Someone has kindly uploaded the official dataset in his name [here](https://github.com/polo5/LDEQ_RwR/issues/2). I made sure this is the same version I used for the results in the paper

### Option 2 (Hard)
While all videos used in WFLW-V have creative commons licences, nvidia has stricter internal privacy rules.
As such, while I provide an official download link for landmark and bboxes annotations, I can only provide a python script to download and crop the videos yourself from youtube.
This is limiting because some videos may disappear in the future. If you want to preprocess the dataset yourself do the following:

1. Download official bboxes annotations [here](https://drive.google.com/file/d/17r2w3abzUsPlsDfqYOsjGGkTp2nvHxds/view?usp=sharing)
2. Download official landmark annotations [here](https://drive.google.com/file/d/1ITmlgXydTogFa5HkE0NvluKAxA4LCloj/view?usp=sharing)
3. `pip install pytube==12.1.2`
3. Run `python utils/download_WFLW_V.py --output_folder /path/to/WFLW_V --n_processes 16`.

Note that pytube may ask you to sign in to your google account and enter some code when you first run this script, depending on the platform you're using.
If things freeze after entering the code (due to multiprocessing), restart the python script.
You may need to restart this script several times if your connection gets closed by youtube servers.

The folder structure should look like:
```
WFLW_V
      |
       bboxes
            |
            -3rDiJYQ6CQ.npy
            -3uh4B-qDcs.npy
            ...
       landmarks
            |
            -3rDiJYQ6CQ.npy
            -3uh4B-qDcs.npy
            ...
       videos
            |
            -3rDiJYQ6CQ.mp4
            -3uh4B-qDcs.mp4
            ...

```


### WFLW-V inference

1) Download the LDEQ trained weights [here](https://drive.google.com/drive/folders/1Jo4BCSSZTBM4Ms3Q9L7dyN4QktbpEM5e?usp=share_link). This particular model uses the fixed point iteration solver to go faster. These weights are different that the ones above because they were obtained with more augmentations (in particular more crops). This helps performance on WFLW_V but worsens performance on WFLW.
2) Edit/pass in args in `test_LDEQ_WFLW_V.py` and run it. You can uncomment the line `solver.test_all_videos_sequential(RWR=args.rwr, plot=True)` in the `main()` funciton if you want to visualize video predictions + ground truth.

### Expected results
```
video zaZ7U2xLD7Y (NME, NMJ) = (2.15, 157.55)
video z_cyma6t-6g (NME, NMJ) = (1.81, 130.94)
video zTU6bPgXoJQ (NME, NMJ) = (2.61, 130.71)
video zMgB081_4_I (NME, NMJ) = (1.43, 118.30)
video zKM2XPvpWXU (NME, NMJ) = (1.61, 78.80)
video ykUEpsSRW4w (NME, NMJ) = (1.57, 89.26)
video yJjnJxF2Nu0 (NME, NMJ) = (1.67, 93.73)
video xlSVN0a6RsU (NME, NMJ) = (1.82, 104.47)
video xUB8PLLaUug (NME, NMJ) = (1.09, 104.60)
video xL4ejntJVgQ (NME, NMJ) = (2.37, 123.16)
video wvgE_DQsGAQ (NME, NMJ) = (1.77, 101.87)
video wdC9K7UbZ04 (NME, NMJ) = (5.62, 103.32)
video v0rtdfHtsMk (NME, NMJ) = (2.77, 135.55)
video uLajzhzHpYY (NME, NMJ) = (1.99, 100.84)
video uL2q-umr2jI (NME, NMJ) = (1.87, 109.11)
video u7y81CPj4JI (NME, NMJ) = (3.63, 153.10)
video tGYaww6dGOY (NME, NMJ) = (1.87, 101.60)
video sF8oSs5LOsM (NME, NMJ) = (2.00, 124.40)
video qZSqE3sBSCE (NME, NMJ) = (1.84, 117.82)
video ofrderRwYwk (NME, NMJ) = (1.86, 115.13)
video oUH2x4K33Zs (NME, NMJ) = (2.35, 137.27)
video n5w0YtKXjuE (NME, NMJ) = (2.67, 120.01)
video mEgqGSXa8nc (NME, NMJ) = (1.41, 95.50)
video m5mmlg7XAwc (NME, NMJ) = (5.12, 119.52)
video lJdmFQ3U4sM (NME, NMJ) = (1.47, 112.88)
video lGZUPhUJBds (NME, NMJ) = (1.17, 84.14)
video l8A3wxCB7ww (NME, NMJ) = (2.44, 113.41)
video l62B_uBUZ5o (NME, NMJ) = (2.15, 119.95)
video kDJRMirM1G8 (NME, NMJ) = (1.36, 139.80)
video hn5RhmE-Mlw (NME, NMJ) = (1.41, 97.21)
video hgB75hM-79g (NME, NMJ) = (4.15, 164.16)
video h35vi6GV5z4 (NME, NMJ) = (2.65, 139.82)
video grIrXxuhdNE (NME, NMJ) = (1.14, 86.22)
video goO421feRRg (NME, NMJ) = (2.53, 134.01)
video g508cTYSlXs (NME, NMJ) = (1.72, 110.15)
video fiFafHWqC4A (NME, NMJ) = (2.37, 111.00)
video egRmXat6LGA (NME, NMJ) = (2.83, 89.47)
video e_q1EOAE_ws (NME, NMJ) = (1.31, 94.42)
video d20Ub0p_s6U (NME, NMJ) = (2.82, 112.34)
video cnQ93bFaKT0 (NME, NMJ) = (1.54, 108.41)
video c03NZ-p2eVM (NME, NMJ) = (1.24, 94.78)
video bYVb_H3jR3M (NME, NMJ) = (2.96, 110.56)
video bQTtpFoh0Dc (NME, NMJ) = (2.46, 117.61)
video bMWY1-1wAZ8 (NME, NMJ) = (1.77, 105.51)
video b5KtjxnATKs (NME, NMJ) = (1.92, 104.82)
video aWpdRUA52Ak (NME, NMJ) = (1.80, 91.14)
video a-RnHKUS7iQ (NME, NMJ) = (2.03, 120.89)
video ZqtExRhXA7U (NME, NMJ) = (2.00, 113.90)
video ZRR0UefCGmU (NME, NMJ) = (1.63, 104.55)
video ZDUJb1cxI7M (NME, NMJ) = (2.06, 128.33)
video Y2IQGYy08AY (NME, NMJ) = (3.13, 162.70)
video Xf-zrQ7Fo-Y (NME, NMJ) = (1.19, 94.40)
video WYGV3gZwkpo (NME, NMJ) = (1.87, 119.24)
video UDq2vOELi2Q (NME, NMJ) = (1.04, 98.03)
video UAttNav2tug (NME, NMJ) = (2.23, 108.75)
video SvwddDllpBI (NME, NMJ) = (3.78, 172.13)
video Qyu6zF75x58 (NME, NMJ) = (1.92, 98.60)
video PQlnVPLIpzY (NME, NMJ) = (2.33, 118.80)
video OGLPaDP1Ld8 (NME, NMJ) = (2.01, 113.90)
video Mm64gFZv-9Y (NME, NMJ) = (1.28, 95.61)
video MQG5iqh6EZE (NME, NMJ) = (1.63, 108.35)
video MLv9kaMv2C0 (NME, NMJ) = (1.46, 85.15)
video KzrYY_TzXDQ (NME, NMJ) = (2.19, 111.14)
video KuoQvtfsQKQ (NME, NMJ) = (2.49, 168.22)
video KZeA8S0dpBI (NME, NMJ) = (2.23, 101.39)
video HkkwOt4GELQ (NME, NMJ) = (1.86, 120.30)
video HVuu-5xki1s (NME, NMJ) = (1.55, 95.82)
video HQc3gBoNVw4 (NME, NMJ) = (1.81, 98.86)
video GSsUSmlmTCs (NME, NMJ) = (1.48, 73.47)
video GJ03hlehycc (NME, NMJ) = (1.25, 87.37)
video FmW0AfxQS7k (NME, NMJ) = (1.16, 97.76)
video CyZb6JSwcy0 (NME, NMJ) = (3.78, 192.22)
video CX2-cG7HMVo (NME, NMJ) = (1.52, 124.68)
video AgTsKDiJ89Q (NME, NMJ) = (1.89, 134.46)
video 90BEx5b51Y0 (NME, NMJ) = (2.30, 137.07)
video 8iFtjR8HTQ4 (NME, NMJ) = (2.52, 132.73)
video 8XAoSLUGJyw (NME, NMJ) = (1.56, 132.49)
video 8Qp2-MUjiWM (NME, NMJ) = (1.60, 117.67)
video 8GaCP6hhAjQ (NME, NMJ) = (1.71, 115.13)
video 8-XM_xpUWLc (NME, NMJ) = (1.28, 96.86)
video 7atlBG9jboo (NME, NMJ) = (1.83, 94.68)
video 7RdNUl1RExw (NME, NMJ) = (1.70, 164.16)
video 7BLNpEVG8uA (NME, NMJ) = (1.70, 90.36)
video 6jO12oweLe8 (NME, NMJ) = (2.77, 144.92)
video 6ZbinFhR13E (NME, NMJ) = (1.47, 108.23)
video 6N5NghZi2XE (NME, NMJ) = (3.26, 141.77)
video 5pJY5CV_mhs (NME, NMJ) = (1.67, 86.55)
video 58sbYPvJcsk (NME, NMJ) = (2.02, 116.27)
video 3HLm5o18B2Q (NME, NMJ) = (2.49, 113.61)
video 2RmgeuwL4UU (NME, NMJ) = (1.80, 124.23)
video 2GonLuutsWk (NME, NMJ) = (3.67, 112.38)
video 1zwvVcekd90 (NME, NMJ) = (1.77, 98.28)
video 1bH8YpHt8Ts (NME, NMJ) = (1.64, 106.15)
video 1bBhtdZ_aCM (NME, NMJ) = (2.11, 87.62)
video 190xnReCUSs (NME, NMJ) = (3.06, 94.42)
video 0m4WVd6Kaqw (NME, NMJ) = (1.35, 96.34)
video 0jfBeIBRMNQ (NME, NMJ) = (1.54, 86.94)
video -tq-Th5W5GM (NME, NMJ) = (1.08, 100.47)
video yuDnjU_MgWg (NME, NMJ) = (2.07, 130.58)
video yZHzrXQBPzs (NME, NMJ) = (1.86, 102.06)
video y4-Ap9NQ8dQ (NME, NMJ) = (2.76, 144.54)
video xrXAZIpLNJw (NME, NMJ) = (1.31, 104.22)
video wzW3xxuRFJY (NME, NMJ) = (2.67, 96.65)
video wwGObwYmgEs (NME, NMJ) = (2.21, 135.28)
video wsf65RMHOJs (NME, NMJ) = (2.29, 112.65)
video wdH0Ckr_da8 (NME, NMJ) = (1.21, 82.99)
video wUAffE59quw (NME, NMJ) = (2.88, 121.40)
video vk3-VRMaGb4 (NME, NMJ) = (2.45, 101.41)
video vijbbBIarJk (NME, NMJ) = (2.16, 112.23)
video vVX22mAPZ7o (NME, NMJ) = (2.24, 84.30)
video vKxlqw-I3Sg (NME, NMJ) = (2.10, 128.61)
video utIqyYORTKA (NME, NMJ) = (3.83, 120.87)
video uRAnjaG-6F0 (NME, NMJ) = (1.29, 92.66)
video uLJ_lM7APH4 (NME, NMJ) = (1.40, 115.17)
video uEcz8I64Mz0 (NME, NMJ) = (2.32, 97.56)
video u987zBNvf7k (NME, NMJ) = (1.71, 143.44)
video tYaWLWBcSLk (NME, NMJ) = (1.39, 98.92)
video t7EdGzaGij0 (NME, NMJ) = (1.44, 111.45)
video stlr8GcjKAo (NME, NMJ) = (2.50, 129.79)
video sWppaW-PBio (NME, NMJ) = (1.90, 105.77)
video rOWzVV_XrcM (NME, NMJ) = (1.29, 84.75)
video rK0_C-Iq-zY (NME, NMJ) = (1.31, 102.00)
video qnIXV3Vvn6A (NME, NMJ) = (1.35, 86.49)
video qYl_863u0QE (NME, NMJ) = (1.31, 103.24)
video qY-LzRPikHY (NME, NMJ) = (2.54, 138.31)
video qVbZpK8XQQQ (NME, NMJ) = (1.55, 110.66)
video pyp-DjbV-OM (NME, NMJ) = (2.08, 132.33)
video pkMI2xACQEM (NME, NMJ) = (2.09, 161.17)
video pdelxvR9sPk (NME, NMJ) = (1.30, 137.53)
video pTt8kLJyPHw (NME, NMJ) = (2.35, 107.53)
video pJKmWfTgH5s (NME, NMJ) = (2.04, 112.87)
video oc7GvC0tpkg (NME, NMJ) = (1.56, 94.49)
video o_Nz9Dg4yzU (NME, NMJ) = (1.42, 91.76)
video ndjdQRJ8nR4 (NME, NMJ) = (1.32, 133.73)
video nZQjlGj1MiA (NME, NMJ) = (1.57, 97.01)
video nEzPYOjjJJM (NME, NMJ) = (1.45, 99.85)
video mqJAPUa__bE (NME, NMJ) = (2.83, 140.79)
video mT0dgSkQxl4 (NME, NMJ) = (1.63, 101.49)
video mChkw5wLlcM (NME, NMJ) = (1.39, 129.46)
video lxs7gEiP_ek (NME, NMJ) = (2.67, 142.08)
video loPBQNpTyEc (NME, NMJ) = (1.82, 143.07)
video lOQxP_GnV6g (NME, NMJ) = (1.91, 101.36)
video l0C66Mx1qy8 (NME, NMJ) = (1.97, 117.04)
video kwZd4vhfe7c (NME, NMJ) = (2.30, 103.51)
video kkFfQO9TaTk (NME, NMJ) = (6.28, 191.40)
video k6cMhiablj8 (NME, NMJ) = (1.49, 84.90)
video jrxBUvygqGU (NME, NMJ) = (4.47, 181.24)
video jZmViWNUoEI (NME, NMJ) = (2.30, 130.63)
video im-rsnl_ZvQ (NME, NMJ) = (1.50, 87.16)
video hSJbF968WfE (NME, NMJ) = (1.60, 145.34)
video hHGtTwO9uZM (NME, NMJ) = (1.98, 116.39)
video gIHx2S65d0s (NME, NMJ) = (3.57, 113.78)
video g8m_lcRcpJA (NME, NMJ) = (1.32, 124.51)
video g8X6FXj0kys (NME, NMJ) = (1.36, 84.06)
video fK39UQ_0B7s (NME, NMJ) = (1.43, 105.59)
video fJQAdyTuB2Q (NME, NMJ) = (1.90, 102.73)
video f7Ca5gnPQSQ (NME, NMJ) = (1.99, 114.17)
video f-lssrQwEGg (NME, NMJ) = (2.40, 120.80)
video etzupT03vEs (NME, NMJ) = (2.55, 100.35)
video en7niHzoPv0 (NME, NMJ) = (1.91, 145.71)
video eXaZUa6YROo (NME, NMJ) = (3.21, 126.30)
video e2RREetKn9g (NME, NMJ) = (2.22, 120.82)
video bfvanngkCWs (NME, NMJ) = (2.21, 123.70)
video bBb2T3TwtuI (NME, NMJ) = (1.80, 114.40)
video aIUXRXablfw (NME, NMJ) = (1.30, 119.87)
video _mAoWlgfat8 (NME, NMJ) = (1.24, 81.07)
video _V4-5J1s6TA (NME, NMJ) = (4.58, 123.74)
video _Skz1SNCsZQ (NME, NMJ) = (1.91, 123.24)
video _QZcLb_hBU8 (NME, NMJ) = (1.46, 108.64)
video _EUvqb9-bJw (NME, NMJ) = (2.53, 141.21)
video _4zbNJmtbGo (NME, NMJ) = (2.32, 105.63)
video _-pq7s8OJok (NME, NMJ) = (1.84, 141.44)
video Ys4SK8W7Sts (NME, NMJ) = (2.03, 234.40)
video Yak0qi1Dy6E (NME, NMJ) = (2.70, 132.29)
video XgrDq23nWI0 (NME, NMJ) = (1.86, 103.96)
video WurAA0n_8rs (NME, NMJ) = (1.01, 82.05)
video US5VmGsXRnE (NME, NMJ) = (1.73, 102.78)
video UIzvn1MXdb0 (NME, NMJ) = (1.31, 113.41)
video U2PU1fOE1NA (NME, NMJ) = (1.49, 82.83)
video TU7XGLc0azQ (NME, NMJ) = (1.74, 152.07)
video TJicKf0pVNg (NME, NMJ) = (1.29, 77.29)
video SxktwXRXk1E (NME, NMJ) = (1.45, 126.89)
video S7g2RHNKkbI (NME, NMJ) = (1.24, 81.93)
video S5dccIR-Ios (NME, NMJ) = (1.10, 84.41)
video RuyXNqmclDc (NME, NMJ) = (2.89, 112.05)
video RqXMZ3pl4tA (NME, NMJ) = (2.06, 154.24)
video Rju1LHxlw4g (NME, NMJ) = (1.67, 79.44)
video Qnj6NpebrF0 (NME, NMJ) = (2.92, 184.16)
video QXl5mmvw2ng (NME, NMJ) = (1.11, 84.97)
video QE3yRpy4LN4 (NME, NMJ) = (1.64, 115.08)
video PrnyqRBsnL0 (NME, NMJ) = (2.27, 144.04)
video Pqmfb9CCnNo (NME, NMJ) = (2.26, 128.39)
video PhFewXluUbc (NME, NMJ) = (1.70, 110.92)
video Okibh3WgOs4 (NME, NMJ) = (2.04, 123.21)
video OkfCLCKPXdw (NME, NMJ) = (1.13, 84.39)
video O5qTmEFFdEA (NME, NMJ) = (1.73, 109.58)
video NwgBDeyztX0 (NME, NMJ) = (2.30, 141.87)
video Nuo-NARd4Hc (NME, NMJ) = (1.76, 137.00)
video Nrc4OSU5V8A (NME, NMJ) = (5.86, 146.37)
video NnOGOECzN5k (NME, NMJ) = (1.70, 108.86)
video NATCW1E011Y (NME, NMJ) = (1.99, 74.09)
video N7FFM5grLeg (NME, NMJ) = (1.66, 91.77)
video N5iuY0JJNmo (NME, NMJ) = (1.58, 87.13)
video MhTZOwsnK8U (NME, NMJ) = (2.08, 116.51)
video MUwU6lfWdwI (NME, NMJ) = (1.44, 104.52)
video MSq9S0NGWF0 (NME, NMJ) = (4.52, 75.30)
video LUbbC3kVAC8 (NME, NMJ) = (3.31, 145.83)
video LN4CwdLUFwI (NME, NMJ) = (2.06, 111.42)
video Kr_X8Bm0Ozw (NME, NMJ) = (2.60, 111.32)
video K_fGufVevRI (NME, NMJ) = (1.13, 73.62)
video KZSefR42hyA (NME, NMJ) = (1.25, 97.79)
video KS35tTtHCfg (NME, NMJ) = (1.98, 123.29)
video KBq0VFzdgXw (NME, NMJ) = (1.81, 115.08)
video JrIo2h_fNkM (NME, NMJ) = (1.89, 168.09)
video JmDZm4jNK4Y (NME, NMJ) = (1.63, 97.03)
video JZW2NzG8xuw (NME, NMJ) = (1.69, 124.00)
video IZ6cSwIE-mQ (NME, NMJ) = (1.36, 114.92)
video IWH0vtA7Cp0 (NME, NMJ) = (2.05, 99.61)
video IUe3PvBM5s8 (NME, NMJ) = (1.71, 111.27)
video I-79nzvXMOE (NME, NMJ) = (1.85, 95.12)
video HIT_ctJ6bJA (NME, NMJ) = (3.61, 114.79)
video GrGI9-ay_sA (NME, NMJ) = (1.62, 87.28)
video G46NHxq8AdU (NME, NMJ) = (2.81, 94.76)
video Fhs7c5KTH0s (NME, NMJ) = (1.86, 108.18)
video FdxdMZgQgJs (NME, NMJ) = (0.96, 214.66)
video FN10IOX1XQc (NME, NMJ) = (3.36, 125.63)
video EoS3YRvdSLw (NME, NMJ) = (2.31, 94.44)
video Eo5WK4rEbXc (NME, NMJ) = (1.73, 109.64)
video Em3NLqTdN2U (NME, NMJ) = (1.70, 94.82)
video EjVeT7tTnV4 (NME, NMJ) = (2.21, 93.38)
video ELkRWC9hEd8 (NME, NMJ) = (2.29, 113.71)
video EKQO1rfhsE4 (NME, NMJ) = (4.34, 154.01)
video DzGB2cLHueU (NME, NMJ) = (2.11, 91.17)
video CGw9onwtZ2E (NME, NMJ) = (2.14, 105.82)
video BsldaqatkWA (NME, NMJ) = (1.61, 107.57)
video B0Q27_-hLnU (NME, NMJ) = (1.69, 122.74)
video Au3-6VIgoH4 (NME, NMJ) = (2.85, 129.70)
video AY0Sh_Ayj7s (NME, NMJ) = (3.09, 107.70)
video AOp0dRNi4AY (NME, NMJ) = (1.48, 96.48)
video 90bCpKDNIPc (NME, NMJ) = (1.59, 100.86)
video 8xM-lsgh71g (NME, NMJ) = (1.53, 98.43)
video 8mj92gpLUig (NME, NMJ) = (1.34, 98.97)
video 8mJGn_Trq2I (NME, NMJ) = (1.82, 141.86)
video 8fBv8W9IfDg (NME, NMJ) = (1.86, 140.61)
video 8MCaEncpPoc (NME, NMJ) = (1.70, 139.34)
video 89OT-prCLZk (NME, NMJ) = (1.13, 138.64)
video 7zRuRbwvEGo (NME, NMJ) = (2.17, 105.51)
video 7jeFPw8Syl0 (NME, NMJ) = (1.94, 70.31)
video 7ijKsN736yo (NME, NMJ) = (2.11, 139.76)
video 7cNnIkKbvWA (NME, NMJ) = (2.03, 152.20)
video 7QWBuSlnhs0 (NME, NMJ) = (2.07, 113.60)
video 7DoMRCru10w (NME, NMJ) = (3.04, 118.75)
video 6V3853n5ISk (NME, NMJ) = (1.45, 104.12)
video 3r6fnb85QQ0 (NME, NMJ) = (1.60, 110.49)
video 3O5j7pbs82A (NME, NMJ) = (3.30, 168.02)
video 2X9l8sHO00k (NME, NMJ) = (1.60, 79.28)
video 2VLEXBOLUnk (NME, NMJ) = (1.72, 133.79)
video 1WYCe3OwXt4 (NME, NMJ) = (1.70, 128.36)
video 1Mb0nu5N_dI (NME, NMJ) = (2.22, 147.45)
video 1JQMo0iRcjY (NME, NMJ) = (2.30, 128.69)
video 1G9rZus2i8A (NME, NMJ) = (1.29, 95.36)
video 1FzWzVapVhM (NME, NMJ) = (1.61, 92.59)
video 1Al9v6hfRs4 (NME, NMJ) = (3.35, 121.26)
video 0v04lIROESw (NME, NMJ) = (1.64, 116.78)
video 0QY9cT3sR_I (NME, NMJ) = (1.00, 110.60)
video 04mtytfl5-0 (NME, NMJ) = (1.62, 130.47)
video sZT43qtJ5h0 (NME, NMJ) = (3.13, 191.24)
video EftDEYjKFj8 (NME, NMJ) = (2.12, 141.50)
video COWFq1Uu3XU (NME, NMJ) = (1.83, 99.15)
video 3Wcq5yYDQqA (NME, NMJ) = (1.51, 105.63)
video 2740z5DLe4E (NME, NMJ) = (1.95, 122.22)
video 8E3Ac3oFG4I (NME, NMJ) = (1.66, 105.60)
video zYinhI5Jn6A (NME, NMJ) = (1.23, 95.28)
video ybWc9Iw806w (NME, NMJ) = (2.08, 100.07)
video yEYKV-RYiI4 (NME, NMJ) = (2.29, 96.24)
video xRb6JpARPms (NME, NMJ) = (1.59, 100.18)
video wfUKgw3J8tM (NME, NMJ) = (1.58, 103.07)
video wDv3BVUBbMo (NME, NMJ) = (1.27, 109.42)
video w6wV7HDwt5A (NME, NMJ) = (3.42, 78.30)
video voRoDG97Oms (NME, NMJ) = (2.64, 203.02)
video vdaSCgwJJKI (NME, NMJ) = (1.38, 116.52)
video vRv5nONV090 (NME, NMJ) = (2.79, 128.08)
video uSeYGIswgWY (NME, NMJ) = (2.64, 165.47)
video uRMY12DEt4M (NME, NMJ) = (1.52, 91.68)
video u4-RvWYmV84 (NME, NMJ) = (1.56, 125.93)
video tn5sEhRHjN4 (NME, NMJ) = (2.01, 110.64)
video sVlIdCWUngI (NME, NMJ) = (1.44, 100.32)
video rH2C2EylVIo (NME, NMJ) = (1.51, 93.31)
video qspPX8Ah__c (NME, NMJ) = (1.75, 121.38)
video qUjnZnYPolw (NME, NMJ) = (1.53, 112.45)
video qIv5OzquVmU (NME, NMJ) = (2.15, 131.54)
video q1fEjrRz1BE (NME, NMJ) = (1.11, 86.36)
video pSzxhhnP8yA (NME, NMJ) = (1.58, 123.86)
video pSWLJN9BpQM (NME, NMJ) = (3.22, 112.72)
video p7amr1ICXCw (NME, NMJ) = (1.06, 88.67)
video p4fRBrwESSA (NME, NMJ) = (1.43, 128.28)
video op0Mbc_QcyQ (NME, NMJ) = (3.13, 150.28)
video oD8Fmf9VZwY (NME, NMJ) = (1.88, 100.85)
video oCil9QHEYyA (NME, NMJ) = (3.03, 113.97)
video ntpxAR5cspU (NME, NMJ) = (2.99, 93.35)
video nXOZ7XUFXyk (NME, NMJ) = (1.27, 122.82)
video nOeAXkUegII (NME, NMJ) = (1.16, 89.72)
video n6f821AA83w (NME, NMJ) = (1.72, 106.75)
video my0MICSp_ro (NME, NMJ) = (1.40, 116.82)
video mKY7n6PSkSQ (NME, NMJ) = (1.29, 142.70)
video kLa347AqeEM (NME, NMJ) = (5.41, 183.66)
video jZFzVP_FggI (NME, NMJ) = (1.27, 95.58)
video jYdxSFVOKFo (NME, NMJ) = (1.99, 122.43)
video jK46D57vuZQ (NME, NMJ) = (1.94, 107.28)
video jDqTUtL-6WM (NME, NMJ) = (2.57, 137.90)
video jAIm6VXo8iY (NME, NMJ) = (1.95, 84.15)
video iy1dOD_x9Rs (NME, NMJ) = (3.13, 99.34)
video iGD-qYAw91M (NME, NMJ) = (3.87, 114.92)
video htcPwawXMIo (NME, NMJ) = (1.48, 105.01)
video hsxXUPn1gPo (NME, NMJ) = (3.33, 111.04)
video hrzhjmX7hP8 (NME, NMJ) = (1.33, 83.96)
video h66xT47iZCc (NME, NMJ) = (1.77, 102.88)
video gLefo_KfymY (NME, NMJ) = (1.78, 117.11)
video gEyuXCPXVwY (NME, NMJ) = (2.75, 171.55)
video gEiJ_ISTfNY (NME, NMJ) = (2.31, 141.06)
video g4xzMj6jyQA (NME, NMJ) = (1.36, 124.20)
video es-O_v_LwI4 (NME, NMJ) = (2.57, 124.80)
video epTXqcYG27s (NME, NMJ) = (1.19, 105.47)
video eEbMsJPwPEM (NME, NMJ) = (2.65, 121.05)
video eA-yWJWkTo8 (NME, NMJ) = (1.25, 99.90)
video dqZKXfitr0Q (NME, NMJ) = (1.46, 103.23)
video cx6qpCfqzKk (NME, NMJ) = (2.54, 131.26)
video cbJXdsK3weI (NME, NMJ) = (2.02, 108.30)
video cXAr9xR_MCE (NME, NMJ) = (1.14, 101.35)
video cRYmM3mAoGI (NME, NMJ) = (1.69, 118.96)
video b7Yd9YaLYPo (NME, NMJ) = (1.53, 129.12)
video apejGuNSeRI (NME, NMJ) = (2.42, 134.92)
video aS-w06SuFCE (NME, NMJ) = (2.43, 101.73)
video aOLhdXG2yuw (NME, NMJ) = (1.57, 94.76)
video _fS0sABXz0g (NME, NMJ) = (3.83, 96.20)
video _QjTU-p9onA (NME, NMJ) = (1.38, 112.08)
video ZzIxTq0nsR4 (NME, NMJ) = (1.41, 127.30)
video ZETfwFDF5XQ (NME, NMJ) = (3.28, 100.56)
video YksHsHdGcEU (NME, NMJ) = (1.61, 116.69)
video YYI4XBLAC9A (NME, NMJ) = (1.64, 116.39)
video XxcB_AOfgG4 (NME, NMJ) = (2.26, 116.11)
video XYcS6xh3Tg4 (NME, NMJ) = (1.55, 116.45)
video X8le6KnrD00 (NME, NMJ) = (1.82, 117.15)
video X70YCtA-cZA (NME, NMJ) = (3.35, 71.73)
video WPGlZH22Uw8 (NME, NMJ) = (1.97, 132.42)
video VraZhV2lu5c (NME, NMJ) = (2.34, 131.64)
video VULz9ol9Ndo (NME, NMJ) = (3.46, 127.71)
video VPN1xFsAbwY (NME, NMJ) = (2.93, 130.34)
video UaNmBaJTWU0 (NME, NMJ) = (1.74, 99.05)
video U6qy45SO7qU (NME, NMJ) = (1.72, 96.05)
video U5jcng1KKWA (NME, NMJ) = (1.31, 97.36)
video U33VSxhm6uc (NME, NMJ) = (1.89, 116.20)
video TycM4H3d5gc (NME, NMJ) = (1.95, 134.44)
video TZV8nrZovJY (NME, NMJ) = (1.50, 87.63)
video SVq5GWaS0jk (NME, NMJ) = (2.22, 94.14)
video RnrOUBzKFWU (NME, NMJ) = (1.68, 109.86)
video ReEasXJDfqo (NME, NMJ) = (1.41, 104.30)
video RWOwPXE6f2A (NME, NMJ) = (1.81, 87.84)
video RC-b5s158uA (NME, NMJ) = (1.37, 111.10)
video R1D5DartKzU (NME, NMJ) = (2.33, 111.22)
video QcQmDCBNrUw (NME, NMJ) = (2.23, 149.65)
video QWMKnvnsx4Y (NME, NMJ) = (1.72, 137.22)
video Q65KjuW1KfE (NME, NMJ) = (1.35, 125.26)
video PCgjHT2jvUc (NME, NMJ) = (2.03, 121.63)
video PBmIhxP7-2w (NME, NMJ) = (1.53, 87.58)
video OaiEbwQx8RU (NME, NMJ) = (1.09, 88.56)
video OAt-hfKxtl0 (NME, NMJ) = (1.00, 83.47)
video NRXtXpd_GqE (NME, NMJ) = (2.47, 162.97)
video N96IDnLJuyo (NME, NMJ) = (1.15, 85.02)
video MrdPFQEc4X0 (NME, NMJ) = (2.34, 98.21)
video MdezOLUKF0Y (NME, NMJ) = (1.58, 113.66)
video L8wkWS1mUL0 (NME, NMJ) = (1.72, 146.16)
video L7aRbYVyaNM (NME, NMJ) = (1.46, 95.61)
video Jez-qmZdMfc (NME, NMJ) = (1.52, 106.79)
video JLQSzww01J0 (NME, NMJ) = (1.14, 87.58)
video HmaE2qgViuo (NME, NMJ) = (1.39, 78.95)
video HAMGlh-T-_w (NME, NMJ) = (1.24, 104.61)
video H1O2qGrw5Tw (NME, NMJ) = (3.12, 132.48)
video Glu_ph5LCNA (NME, NMJ) = (1.80, 118.85)
video FeKTr_R-XX0 (NME, NMJ) = (1.49, 97.39)
video FUSoIU-vc30 (NME, NMJ) = (3.34, 216.00)
video Esb1PdgVOgE (NME, NMJ) = (2.39, 107.75)
video ENd5jQGc0Z8 (NME, NMJ) = (1.97, 142.77)
video DfQ3sskMCE4 (NME, NMJ) = (1.73, 126.44)
video DTRvtX974k0 (NME, NMJ) = (1.73, 111.45)
video DEB6oACyIfE (NME, NMJ) = (1.19, 96.58)
video D9tNYcI0WOM (NME, NMJ) = (1.79, 96.16)
video D0LtQYsVIKI (NME, NMJ) = (2.94, 156.78)
video Cpsd-bB9P9Y (NME, NMJ) = (1.31, 108.36)
video BLby_ZAVARs (NME, NMJ) = (2.15, 106.83)
video AvPsZ8tByRA (NME, NMJ) = (1.16, 105.81)
video AsHfglE8Ryk (NME, NMJ) = (1.40, 89.05)
video AQfiRM26Hts (NME, NMJ) = (9.45, 98.86)
video ADRWkhUA9LE (NME, NMJ) = (1.47, 132.25)
video A32jtzx43fE (NME, NMJ) = (3.97, 123.11)
video 9SD9Pl6ijKk (NME, NMJ) = (1.63, 98.20)
video 9P4Gtll4MtM (NME, NMJ) = (1.05, 122.74)
video 8zZMdzafMRQ (NME, NMJ) = (1.18, 103.33)
video 8vu2ynxs3l8 (NME, NMJ) = (1.47, 128.65)
video 8iqkbFM0EHg (NME, NMJ) = (2.46, 118.52)
video 8fUDbr_9qpA (NME, NMJ) = (1.79, 125.25)
video 8dAT-x_h_Kg (NME, NMJ) = (2.95, 208.41)
video 7w-ZMzhoJTQ (NME, NMJ) = (1.45, 98.75)
video 7isjQJT5c0k (NME, NMJ) = (2.32, 154.28)
video 6zqin_VL9YM (NME, NMJ) = (1.38, 101.14)
video 6tvlyy2ux74 (NME, NMJ) = (1.53, 88.59)
video 6KS55pjfsjs (NME, NMJ) = (1.78, 88.62)
video 6Go_FG8tKyQ (NME, NMJ) = (1.67, 115.54)
video 68TYk6i_Nyc (NME, NMJ) = (1.69, 99.89)
video 5Xc1Loj627A (NME, NMJ) = (1.69, 117.62)
video 5W1mdd26rCs (NME, NMJ) = (1.53, 122.37)
video 5JYMFl7wuYE (NME, NMJ) = (1.61, 107.17)
video 5BWctpxOQ14 (NME, NMJ) = (2.12, 124.35)
video 5AutD0Q1sDk (NME, NMJ) = (1.39, 108.37)
video 3jnBOzDc-Mk (NME, NMJ) = (2.55, 140.36)
video 3B2C_CAds2U (NME, NMJ) = (1.50, 96.81)
video 2uLZ52gkpZQ (NME, NMJ) = (1.17, 103.15)
video 2-6VPFkFeMY (NME, NMJ) = (1.39, 113.49)
video 1kc9Y4nLiW0 (NME, NMJ) = (1.57, 101.15)
video 13NQSzojE3k (NME, NMJ) = (3.05, 99.98)
video 03w12cEmLK8 (NME, NMJ) = (4.03, 127.24)
video -wH4eCTOc_Y (NME, NMJ) = (1.76, 110.48)
video -fOhqLVO4n0 (NME, NMJ) = (1.87, 92.71)
video -Ah8p7B2vvM (NME, NMJ) = (1.68, 121.86)
video H1Asad4TZcU (NME, NMJ) = (1.20, 100.30)
video z4FG92f4IA0 (NME, NMJ) = (3.12, 120.52)
video gQ9CRQFXF7k (NME, NMJ) = (1.63, 74.00)
video fvH4r-lNih8 (NME, NMJ) = (3.03, 101.86)
video bDU--ZZ4GAE (NME, NMJ) = (3.72, 120.48)
video aEDGkJTVinc (NME, NMJ) = (2.31, 95.60)
video YTSpV-DFgcA (NME, NMJ) = (3.06, 110.58)
video UT_lU6VxmpE (NME, NMJ) = (1.86, 124.25)
video QywGFCkIIA8 (NME, NMJ) = (1.24, 122.96)
video PbMhpoQPztc (NME, NMJ) = (1.25, 108.97)
video KQF2f2pXu7U (NME, NMJ) = (2.19, 134.98)
video KONm6uviFY4 (NME, NMJ) = (2.42, 130.93)
video KGTY1x48goQ (NME, NMJ) = (2.45, 109.17)
video GUIDMMJPv0s (NME, NMJ) = (2.87, 90.22)
video Dch5j3KmQ_w (NME, NMJ) = (2.04, 104.84)
video A7Togt8UT5I (NME, NMJ) = (2.15, 128.42)
video 8d5X3CPVU6o (NME, NMJ) = (1.47, 95.57)
video 6DO_XtUNQRI (NME, NMJ) = (1.80, 97.09)
video zEdAA9Oxpdw (NME, NMJ) = (1.52, 88.38)
video yhte-FNCmSg (NME, NMJ) = (2.23, 126.03)
video uiDQd3d9Wlc (NME, NMJ) = (1.56, 81.87)
video sb_8ucyTXn0 (NME, NMJ) = (1.51, 91.56)
video sY1RIYLIYoM (NME, NMJ) = (1.67, 131.27)
video qUoRIBHGBgc (NME, NMJ) = (1.39, 129.62)
video p_DN1I8lNNE (NME, NMJ) = (1.45, 96.97)
video pDNGWgZwOM8 (NME, NMJ) = (1.80, 102.93)
video nJAr8gGbvnY (NME, NMJ) = (1.26, 91.94)
video nB8kO8q6WAs (NME, NMJ) = (2.22, 94.21)
video liOfG3g5DdY (NME, NMJ) = (2.40, 133.11)
video hswX8HTHbc8 (NME, NMJ) = (1.52, 106.05)
video gscMBIsCf7E (NME, NMJ) = (1.48, 122.86)
video g56sKvNWGk0 (NME, NMJ) = (2.13, 128.16)
video f6n2OyGIKEE (NME, NMJ) = (2.54, 93.29)
video f5QWD-LENsA (NME, NMJ) = (2.12, 114.05)
video eUk6PMcw998 (NME, NMJ) = (1.15, 104.74)
video cVH77o7y-9w (NME, NMJ) = (1.73, 122.98)
video bcn2sFUSQ3Q (NME, NMJ) = (2.22, 135.23)
video bYpRqB7Lz00 (NME, NMJ) = (1.46, 117.05)
video aMUzZt1OWC4 (NME, NMJ) = (1.53, 95.61)
video a2a5QmwDu7k (NME, NMJ) = (2.05, 118.68)
video _wR49qyHRhM (NME, NMJ) = (1.59, 194.74)
video Yj07sVp_WwM (NME, NMJ) = (1.73, 120.59)
video XYwoZiUir7w (NME, NMJ) = (1.56, 111.44)
video WxpsHlR_izQ (NME, NMJ) = (1.65, 105.25)
video WHi86CIyEdw (NME, NMJ) = (2.40, 111.50)
video W8kTIEAARCo (NME, NMJ) = (2.46, 123.24)
video VbtpoIENJOw (NME, NMJ) = (3.59, 115.65)
video U_7n5U5J52g (NME, NMJ) = (1.87, 94.72)
video Sa9uxTy0EgA (NME, NMJ) = (1.78, 91.09)
video RGUNha2Jkss (NME, NMJ) = (1.64, 124.21)
video RFucOnEcacE (NME, NMJ) = (1.37, 101.42)
video QLtXCqnE34c (NME, NMJ) = (1.71, 121.01)
video PISip-Pp_z4 (NME, NMJ) = (2.09, 159.47)
video Nt3IZ0yf6Hg (NME, NMJ) = (1.64, 134.27)
video NFtkkuqNvMY (NME, NMJ) = (1.29, 113.67)
video Lm9ssQU9vik (NME, NMJ) = (1.53, 129.34)
video LLr0lOF685M (NME, NMJ) = (1.47, 107.96)
video K3DiFz-ifSE (NME, NMJ) = (2.56, 154.29)
video GZ-twlzax5Y (NME, NMJ) = (2.15, 141.58)
video GQVwpZf-d_Y (NME, NMJ) = (2.16, 114.69)
video EyzJ4wVr8JI (NME, NMJ) = (4.88, 210.40)
video EkeIL6ZxxKY (NME, NMJ) = (1.95, 113.06)
video DdVrtoOWMnY (NME, NMJ) = (1.24, 114.14)
video DFRt9_LeGVU (NME, NMJ) = (1.64, 91.47)
video AjcKi__RaMA (NME, NMJ) = (2.36, 142.88)
video ARUwWDtMnHc (NME, NMJ) = (1.89, 114.63)
video 98voIiSm2XE (NME, NMJ) = (2.27, 124.14)
video 8bxwa3rbgVI (NME, NMJ) = (1.53, 98.91)
video 7fOGxY6Nbkg (NME, NMJ) = (1.77, 160.30)
video 7YqdeV2IRrI (NME, NMJ) = (2.19, 111.01)
video 5s-WZcNYhLU (NME, NMJ) = (1.30, 97.11)
video 4Lq3AhIsc80 (NME, NMJ) = (1.53, 129.60)
video 3HartDFoBaI (NME, NMJ) = (1.47, 134.71)
video 2Qznc5WzRI8 (NME, NMJ) = (2.76, 91.96)
video -RXLf5WTLjw (NME, NMJ) = (2.53, 175.33)
video -IX0CnTHWnc (NME, NMJ) = (3.02, 172.27)
```
```
Avg (NME, NMJ) = 2.02, 115.48
```

## Training
nvidia won't release it, sorry:(

## Cite

If you find this work useful, please consider citing us:

```
@InProceedings{Micaelli_2023_CVPR,
    author    = {Micaelli, Paul and Vahdat, Arash and Yin, Hongxu and Kautz, Jan and Molchanov, Pavlo},
    title     = {Recurrence Without Recurrence: Stable Video Landmark Detection With Deep Equilibrium Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {22814-22825}
}
```
