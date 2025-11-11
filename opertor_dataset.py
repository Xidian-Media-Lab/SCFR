import os
import cv2

Root_Dir = "./dataset/Vox2-mp4/dev"

def is_clear_image(img, threshold=15):
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var <= threshold# 阈值越高，越清晰

for root_dir in os.listdir(Root_Dir):
    root_dir = Root_Dir +'/' + root_dir
    for files in os.listdir(root_dir):
        files = root_dir + '/' + files
        for video in os.listdir(files):
            video_path = files + '/' + video
            cap = cv2.VideoCapture(video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            not_clear_video = None
            for i in range(0, total_frames, 30):
                if i == 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    _, source_frame = cap.read()
                    source_frame = cv2.resize(source_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    if is_clear_image(source_frame):
                        not_clear_video = video_path
                        break
                    source_frame_name = f"frame_{i:04d}.png"
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    _, driving_frame = cap.read()
                    driving_frame = cv2.resize(driving_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    driving_frame_name = f"frame_{i:04d}.png"

                    makedir_name = files + '/' + video.split('.mp4')[0]
                    if not os.path.exists(makedir_name):
                        os.makedirs(makedir_name)
                    frame_path = makedir_name + '/' + f"{i:04d}"
                    if not os.path.exists(frame_path):
                        os.makedirs(frame_path)

                    cv2.imwrite(frame_path+ '/' + source_frame_name, source_frame)
                    cv2.imwrite(frame_path + '/' + driving_frame_name, driving_frame)
            print(f"Finish {video_path}, direct_remove {not_clear_video}")
            os.remove(video_path)
        if os.path.exists(files) and not any(os.scandir(files)):
            os.rmdir(files)
            print(f"{files} 已被删除")
