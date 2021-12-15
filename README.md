# minDALL-E on Conceptual Captions

`minDALL-E`, named after [minGPT](https://github.com/karpathy/minGPT), is a 1.3B text-to-image generation model trained on 14 million
image-text pairs for non-commercial purposes.

![a painting of a bird in the style of asian painting](assets/bird_asian_painting_style.gif)
![a photo of san francisco's golden gate bridge in black and white tone](assets/golden_gate_black_and_white_tone.gif)


## Environment Setup
- Basic setup
```
PyTorch == 1.8.0
CUDA >= 10.1
```
- Other packages
```
pip install -r requirements.txt
```

## Model Checkpoint
- Model structure (two-stage autoregressive model)
  - Stage1: Unlike the original DALL-E [1], we replace Discrete VAE with VQGAN [2] to generate high-quality samples effectively.
            We slightly fine-tune [vqgan_imagenet_f16_16384](https://github.com/CompVis/taming-transformers), provided by the official VQGAN repository, on FFHQ [3] as well as ImageNet.
  - Stage2: We train our 1.3B transformer from scratch on 14 million image-text pairs from CC3M [4] and CC12M [5]. For the more detailed model spec, please see [configs/dalle-1.3B.yaml](configs/dalle-1.3B.yaml).
- You can download the pretrained models including the tokenizer from [this link](https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz). This will require about 5GB space.

## Sampling
- Given a text prompt, the code snippet below generates candidate images and re-ranks them using OpenAI's CLIP [6].
- This has been tested under a single V100 of 32GB memory. In the case of using GPUs with limited memory, please lower down num_candidates to avoid OOM.
```python
from matplotlib import pyplot as plt
import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score

device = 'cuda:0'
set_seed(0)

prompt = "A painting of a monkey with sunglasses in the frame"
model = Dalle.from_pretrained('minDALL-E/1.3B')  # This will automatically download the pretrained model.
model.to(device=device)

# Sampling
images = model.sampling(prompt=prompt,
                        top_k=256, # It is recommended that top_k is set lower than 256.
                        top_p=None,
                        softmax_temperature=1.0,
                        num_candidates=96,
                        device=device).cpu().numpy()
images = np.transpose(images, (0, 2, 3, 1))

# CLIP Re-ranking
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)
model_clip.to(device=device)
rank = clip_score(prompt=prompt,
                  images=images,
                  model_clip=model_clip,
                  preprocess_clip=preprocess_clip,
                  device=device)

# Plot images
images = images[rank]
plt.imshow(images[0])
plt.show()
```
- If you want to use a complete python code for sampling, please see [examples/sampling_ex.py](examples/sampling_ex.py)
- If you want to play with an interactive demo, please see [examples/sampling_interactive_demo.ipynb](examples/sampling_interactive_demo.ipynb).
  Before using this, you may need to install [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html).

## Samples (Top-K=256, Temperature=1.0)
- "a painting of a {cat, dog} with sunglasses in the frame"
<p float="left">
  <img src="/assets/a painting of a cat with sunglasses in the frame_0.png" width="128" />
  <img src="/assets/a painting of a cat with sunglasses in the frame_1.png" width="128" />
  <img src="/assets/a painting of a cat with sunglasses in the frame_2.png" width="128" />
  <img src="/assets/a painting of a cat with sunglasses in the frame_3.png" width="128" />
  <img src="/assets/a painting of a cat with sunglasses in the frame_4.png" width="128" />
  <img src="/assets/a painting of a cat with sunglasses in the frame_5.png" width="128" />
</p>
<p float="left">
  <img src="/assets/a painting of a dog with sunglasses in the frame_0.png" width="128" />
  <img src="/assets/a painting of a dog with sunglasses in the frame_1.png" width="128" />
  <img src="/assets/a painting of a dog with sunglasses in the frame_2.png" width="128" />
  <img src="/assets/a painting of a dog with sunglasses in the frame_3.png" width="128" />
  <img src="/assets/a painting of a dog with sunglasses in the frame_4.png" width="128" />
  <img src="/assets/a painting of a dog with sunglasses in the frame_5.png" width="128" />
</p>

- "a large {pink, black} elephant walking on the beach"
<p float="left">
  <img src="/assets/A large pink elephant walking on the beach_0.png" width="128" />
  <img src="/assets/A large pink elephant walking on the beach_1.png" width="128" />
  <img src="/assets/A large pink elephant walking on the beach_2.png" width="128" />
  <img src="/assets/A large pink elephant walking on the beach_3.png" width="128" />
  <img src="/assets/A large pink elephant walking on the beach_4.png" width="128" />
  <img src="/assets/A large pink elephant walking on the beach_5.png" width="128" />
</p>
<p float="left">
  <img src="/assets/A large black elephant walking on the beach_0.png" width="128" />
  <img src="/assets/A large black elephant walking on the beach_1.png" width="128" />
  <img src="/assets/A large black elephant walking on the beach_2.png" width="128" />
  <img src="/assets/A large black elephant walking on the beach_3.png" width="128" />
  <img src="/assets/A large black elephant walking on the beach_4.png" width="128" />
  <img src="/assets/A large black elephant walking on the beach_5.png" width="128" />
</p>

- "Eiffel tower on a {desert, mountain}"
<p float="left">
  <img src="/assets/Eiffel tower on a desert_0.png" width="128" />
  <img src="/assets/Eiffel tower on a desert_1.png" width="128" />
  <img src="/assets/Eiffel tower on a desert_2.png" width="128" />
  <img src="/assets/Eiffel tower on a desert_3.png" width="128" />
  <img src="/assets/Eiffel tower on a desert_4.png" width="128" />
  <img src="/assets/Eiffel tower on a desert_5.png" width="128" />
</p>
<p float="left">
  <img src="/assets/Eiffel tower on a mountain_0.png" width="128" />
  <img src="/assets/Eiffel tower on a mountain_1.png" width="128" />
  <img src="/assets/Eiffel tower on a mountain_2.png" width="128" />
  <img src="/assets/Eiffel tower on a mountain_3.png" width="128" />
  <img src="/assets/Eiffel tower on a mountain_4.png" width="128" />
  <img src="/assets/Eiffel tower on a mountain_5.png" width="128" />
</p>

## Quantitative Results
* We have validated `minDALL-E` on the CC3M validation set (in-distribution evaluation) and MS-COCO (zero-shot evaluation).
* For CC3M, we measure the cosine similarity between image and text representations from the pretrained CLIP model (ViT-B/32), referred to as CLIP-score.
* For MS-COCO, we compute FID between 30K generated and real samples from MS-COCO 2017, where we randomly choose 30K captions from COCO as in DALL-E.
  We select the best out of 32 candidates by CLIP re-ranking.

| Model | CC3M:CLIP-score (higher is better) | MS-COCO:FID-30K (lower is better) |
|:------|----:|----:|
|VQGAN [2]    | 0.20 | -    |
|ImageBART [7]| 0.23 | -    |
|DALL-E [1]   | -    | 27.5 |
|minDALL-E    | **0.26** | **14.7** |


## Transfer Learning Examples
* `minDALL-E`, which is pre-trained on noisy text supervisions, could be transferable to class-conditional and unconditional generation tasks. To validate this, we simply fine-tune it on ImageNet over 8 epochs in the case of [class-conditional generation](configs/transfer-imagenet-clscond-gen.yaml) and [unconditional generation](configs/transfer-imagenet-uncond-gen.yaml).
* The commands below fine-tune the pretrained DALL-E. It takes about 36 hours on 8 V100 GPUs.
```
# unconditinoal image generation for imagenet (256x256)
python examples/transfer_learning_ex.py -d=configs/transfer-imagenet-uncond-gen.yaml
                                        -u=[MODEL_CKPT]
                                        -r=[RESULT_PATH]
                                        --n-gpus=[NUM_GPUS]

# class-conditinoal image generation for imagenet (256x256)
python examples/transfer_learning_ex.py -d=configs/transfer-imagenet-clscond-gen.yaml
                                        -u=[MODEL_CKPT]
                                        -r=[RESULT_PATH]
                                        --n-gpus=[NUM_GPUS]
```
* We compute FID-50K between 50K generated samples and all ImageNet training samples, where we use top-k=256 and softmax temperature=1.0 for generation.
  All results are obtained without the rejection sampling. Interestingly, our model achieves very competitive performance with baselines, even though `minDALL-E` is fine-tuned in a few epochs.

| Model | Params | FID-50K(class-cond.) | FID-50K(uncond.) |
|:-----|----:|----:|----:|
|VQ-GAN    | 1.4B | 15.78 | - |
|ImageBART | 3.5B | 21.19 | - |
|minDALL-E | 1.3B | **15.55** | 37.58 |


## BibTex
If you find this repository useful in your research, please cite:
```
@misc{kakaobrain2021minDALL-E,
  title         = {minDALL-E on Conceptual Captions},
  author        = {Saehoon Kim, Sanghun Cho, Chiheon Kim, Doyup Lee, and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/minDALL-E}},
}
```

## References
* [1] Ramesh et al. Zero-Shot Text-to-Image Generation, ICML 2021.
* [2] Esser et al. Taming Transformers for High-Resolution Image Synthesis, CVPR 2021.
* [3] Karras et al. A Style-Based Generator Architecture for Generative Adversarial Networks, CVPR 2019.
* [4] Sharma et al. Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning, ACL 2018.
* [5] Changpinyo et al. Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts, CVPR 2021.
* [6] Radford et al. Learning Transferable Visual Models From Natural Language Supervision, ICML 2021.
* [7] Esser et al. ImageBART: Bidirectional Context with Multinomial Diffusion for Autoregressive Image Synthesis, NeurIPS 2021.
* [8] https://github.com/karpathy/minGPT


## Licenses
* The `source codes` are licensed under [Apache 2.0](LICENSE.apache-2.0) License.
* The `stage2 pretrained weights` are licensed under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License.

## Contact
We hope that `minDALL-E` helps various projects in research-oriented institutes and startups.
If you would like to collaborate with us or share a feedback, please e-mail to us, contact@kakaobrain.com

## Limitations
Although `minDALL-E` is trained on a small set (14M image-text pairs), this might be vulnerable to malicious attacks from the prompt engineering to generate socially unacceptable images. If you obersve these images, please report the "prompt" and "generated images" to us.
