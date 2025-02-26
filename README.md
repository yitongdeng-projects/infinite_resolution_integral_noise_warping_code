# [ICLR 2025] Infinite-Resolution Integral Noise Warping for Diffusion Models
by [Yitong Deng](https://yitongdeng.github.io/), [Winnie Lin](https://web.stanford.edu/~wl1915/), [Lingxiao Li](https://people.csail.mit.edu/lingxiao/), [Dmitriy Smirnov](https://dsmirnov.me/), [Ryan Burgert](https://ryanndagreat.github.io/), [Ning Yu](https://ningyu1991.github.io/), Vincent Dedun, Mohammad H. Taghavi.

Paper: [OpenReview](https://openreview.net/forum?id=Y6LPWBo2HP) / [Arxiv](https://arxiv.org/abs/2411.01212)

## Overview
On a high level, our code takes as input flow map data (e.g. optical flow) and outputs white noise images that are warped / advected by the input flow map. These white noise frames are spatially uncorrelated (thus making them suitable for image diffusion models) and temporally correlated (thus adding in cross-frame consistency). When playing back our warped noise images as a video, the noise video should show the same motion as the original video from which the flow is extracted; but when paused at any given frame, one should not be able to tell it apart form a randomly sampled white noise image.

## Run
The only "installation" required to run our code is to pip install taichi. 
```bash
pip install taichi
```
Once taichi is installed, simply run:
```bash
python test.py -n [exp_name]
```
We assume that a folder `data/[exp_name]` exists and contains a file named `flows.npy` with a `[num_frames, H, W, 2]` array of flow map data. To get you started, we include three sample flow map sequences: `bear`, `lucia`, and `motorbike` along with the original videos from which they are extracted. 
When testing on your own flow maps, please make sure to reshape it accordingly. We assume that the 2-vector `(r, c)` stored in each entry of the input array has conventions `r` going from top to bottom and `c` going from left to right.   

We tested our code on `Windows 11` with `CUDA 11.8`, `Python 3.10.9`, and `Taichi 1.7.3`'.

## Output
The warped noise will be saved as `warped_noise.npy` and a sequence of `.jpg` images for visualization in `logs/[exp_name]/particle`.

## Bibliography
If you find our paper or code helpful, consider citing:
```
@misc{deng2024infiniteresolutionintegralnoisewarping,
      title={Infinite-Resolution Integral Noise Warping for Diffusion Models}, 
      author={Yitong Deng and Winnie Lin and Lingxiao Li and Dmitriy Smirnov and Ryan Burgert and Ning Yu and Vincent Dedun and Mohammad H. Taghavi},
      year={2024},
      eprint={2411.01212},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.01212}, 
}
```
