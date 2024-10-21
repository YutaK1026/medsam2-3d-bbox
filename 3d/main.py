import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

CHECKPOINT = "../model/3d/latest_epoch.pth"
CONFIG = "sam2_hiera_s.yaml"
VIDEO_FRAMES_DIRECTORY_PATH = "../video_data/data"
sam2_model = build_sam2_video_predictor(CONFIG, CHECKPOINT)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def get_bbox(mask):
    """
    mask_list:
        ([すべて0],[x座標],[y座標])
    segmentationした座標が入力されている
    大体円とかそんな形になっているはず
    これをすべて内包するbboxを生成したい -> x, yそれぞれの最大最小を出力すればよいのではないか
    """
    mask_list = np.where(mask)  # 元はmask == Trueだった．flakeに怒られてこうした
    y_max = np.amax(mask_list[1])
    y_min = np.amin(mask_list[1])
    x_max = np.amax(mask_list[2])
    x_min = np.amin(mask_list[2])
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float64).tolist()
    return box


def segment_with_bbox(
    frame_names: list[str],
    start_index: int,
    end_index: int,
    obj_id: int,
    box: np.ndarray,
):
    inference_state = sam2_model.init_state(video_path=VIDEO_FRAMES_DIRECTORY_PATH)
    sam2_model.reset_state(inference_state)

    _, out_obj_ids, out_mask_logits = sam2_model.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=start_index,
        obj_id=obj_id,
        box=box,
    )
    # bboxを付与

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    vis_frame_stride = 1
    plt.close("all")
    new_list = []
    for out_frame_idx in range(start_index, end_index, vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(
            Image.open(
                os.path.join(VIDEO_FRAMES_DIRECTORY_PATH, frame_names[out_frame_idx])
            )
        )
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            if True in out_mask:
                dict = {}
                dict["id"] = out_frame_idx
                dict["box"] = get_bbox(out_mask)
                new_list.append(dict)
            show_mask(out_mask, plt.gca(), obj_id=1)
        plt.savefig(f"./datas/data_009/image_{out_frame_idx}.png")
    json.dump(new_list, dict_json)


dict_json = open("datas/data.json", "w")

frame_names = [
    p
    for p in os.listdir(VIDEO_FRAMES_DIRECTORY_PATH)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
box = np.array([160, 313, 174, 323], dtype=np.float32)

segment_with_bbox(
    frame_names=frame_names,
    start_index=52,
    end_index=len(frame_names),
    obj_id=None,
    box=box,
)

# # bboxラベルが付与されたものより上のsegmentationを行う
# frame_names = frame_names[0:114]
# frame_names.reverse()
# segment_with_bbox(
#     frame_names=frame_names, start_index=0, end_index=114, obj_id=4, box=box
# )
