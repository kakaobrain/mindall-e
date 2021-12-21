"""
download the weights to 'pretrained' first
wget https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz
tar -xvf 1.3B.tar.gz
"""

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cog
import clip
from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score


class Predictor(cog.Predictor):
    def setup(self):
        self.device = 'cuda:0'
        self.model = Dalle.from_pretrained("pretrained/1.3B")
        self.model.to(device=self.device)
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)
        self.model_clip.to(device=self.device)

    @cog.input(
        "prompt",
        type=str,
        help="Prompt for generating image",
    )
    @cog.input(
        "num_samples",
        type=int,
        default=1,
        min=1,
        max=9,
        help="Number of generated images.",
    )
    @cog.input(
        "seed",
        type=int,
        default=0,
        help="Set seed. 0 for random seed.",
    )
    def predict(self, prompt, num_samples, seed):
        softmax_temperature = 1
        top_k = 256
        num_candidates = 30
        set_seed(seed)

        images = self.model.sampling(prompt=prompt,
                                     top_k=top_k,
                                     top_p=None,
                                     softmax_temperature=softmax_temperature,
                                     num_candidates=num_candidates,
                                     device=self.device).cpu().numpy()
        images = np.transpose(images, (0, 2, 3, 1))

        # CLIP Re-ranking
        rank = clip_score(prompt=prompt, images=images, model_clip=self.model_clip,
                          preprocess_clip=self.preprocess_clip, device=self.device)
        images = images[rank]

        images = images[:num_samples]

        images = [Image.fromarray((images[i] * 255).astype(np.uint8)) for i in range(num_samples)]
        grid = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3)}
        res = concat_images(images, (256, 256), grid[num_samples])
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        res.save(str(out_path))
        return out_path


def concat_images(images, size, shape=None):
    width, height = size
    images = [ImageOps.fit(image, size, Image.ANTIALIAS)
              for image in images]

    # Create canvas for the final image with total size
    shape = shape if shape else (len(images), 1)
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('RGB', image_size, color=(255, 255, 255))

    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            if idx < len(images):
                image.paste(images[idx], offset)

    return image
