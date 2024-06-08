# 3D Gaussian Splatting as Markov Chain Monte Carlo


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{kheradmand20243d,
  title={3D Gaussian Splatting as Markov Chain Monte Carlo},
  author={Kheradmand, Shakiba and Rebain, Daniel and Sharma, Gopal and Sun, Weiwei and Tseng, Jeff and Isack, Hossam and Kar, Abhishek and Tagliasacchi, Andrea and Yi, Kwang Moo},
  journal={arXiv preprint arXiv:2404.09591},
  year={2024}
}</code></pre>
  </div>
</section>



## How to install

Please refer to [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) for installation instructions. Make sure to reinstall diff-gaussian-rasterization with the following command on an available 3DGS environment as this library has been modified:
```
pip install submodules/diff-gaussian-rasterization
```

## How to run
Running code is similar to the [Original 3DGS code base](https://github.com/graphdeco-inria/gaussian-splatting) with the following differences:
- You need to specify the maximum number of Gaussians that will be used. This is performed using --cap_max argument. The results in the paper uses the final number of Gaussians reached by the original 3DGS run for each shape.
- You need to specify the scale regularizer coefficient. This is performed using --scale_reg argument. For all the experiments in the paper, we use 0.01.
- You need to specify the opacity regularizer coefficient. This is performed using --opacity_reg argument. For Deep Blending dataset, we use 0.001. For all other experiments in the paper, we use 0.01.
- You need to specify the noise learning rate. This is performed using --noise_lr argument. For all the experiments in the paper, we use 5e5.
- You need to specify the initialization type. This is performed using --init_type argument. Options are random (to initialize randomly) or sfm (to initialize using a pointcloud).




