"""
download the weights to 'pretrained' first
wget https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz
tar -xvf 1.3B.tar.gz
"""
from typing import List
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from cog import BasePredictor, Path, Input, BaseModel
import clip

from dalle.models import Dalle
from dalle.utils.utils import set_seed, clip_score


class ModelOutput(BaseModel):
    image: Path


class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda:0'
        self.model = Dalle.from_pretrained("pretrained/1.3B")
        self.model.to(device=self.device)
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=self.device)
        self.model_clip.to(device=self.device)

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for generating image.",
        ),
        num_samples: int = Input(
            default=4,
            ge=1,
            le=9,
            description="Number of generated images.",
        ),
        seed: int = Input(
            default=0,
            description="Set seed. 0 for random seed.",
        ),
    ) -> List[ModelOutput]:
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
        print(type(images[0]))
        output = []
        for i, array in enumerate(images):
            img = Image.fromarray((images[i] * 255).astype(np.uint8))
            output_path = Path(tempfile.mkdtemp()) / f"output_{i}.png"
            img.save(str(output_path))
            img.save(f'hi_{i}.png')
            output.append(ModelOutput(image=output_path))
        return output
