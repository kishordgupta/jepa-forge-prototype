import argparse
from jepa_forge.vision_transfer import run_vision

p=argparse.ArgumentParser()
p.add_argument("--config",default="configs/vision_transfer.json")
p.add_argument("--output",default="artifacts/extension/vision")
p.add_argument("--image-cache",default=".cache/pretrained")
p.add_argument("--video-cache",default=".cache/vjepa")
a=p.parse_args()
run_vision(a.config,a.output,a.image_cache,a.video_cache)
