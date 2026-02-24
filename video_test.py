import torch
import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import sam3
import torch
from PIL import Image
from sam3.visualization_utils import show_box, show_mask, show_points
from sam3.model_builder import build_sam3_video_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


plt.rcParams["axes.titlesize"] = 12
plt.rcParams["figure.titlesize"] = 12

# Load model
print("=" * 12 + "Loading Model" + "=" * 12)
sam3_root = os.getcwd() 
sam3_model = build_sam3_video_model()
sam3_model = sam3_model.half()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone


# Initialize inference state
video_path = f"{sam3_root}/media/videos/penguins.mp4"
inference_state = predictor.init_state(video_path=video_path)

predictor.clear_all_points_in_video(inference_state)

# Turns a video into frames, then cv2 just convets it to RGB. Then releases them when done.
cap = cv2.VideoCapture(video_path)
video_frames_for_vis = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    video_frames_for_vis.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

frame0 = video_frames_for_vis[0]
width, height = frame0.shape[1], frame0.shape[0]



ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (210, 350) to get started
points = np.array([[210, 350]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)

rel_points = [[x / width, y / height] for x, y in points]

points_tensor = torch.tensor(rel_points, dtype=torch.float32)
points_labels_tensor = torch.tensor(labels, dtype=torch.int32)

_, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points_tensor,
    labels=points_labels_tensor,
    clear_old_points=False,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(frame0)
show_points(points, labels, plt.gca())
show_mask((video_res_masks[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()