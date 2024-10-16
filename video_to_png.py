import supervision as sv

frames_generator = sv.get_video_frames_generator("video.mp4")
sink = sv.ImageSink(
    target_dir_path="video_data/data",
    image_name_pattern="{:05d}.png")

with sink:
    for frame in frames_generator:
        sink.save_image(frame)