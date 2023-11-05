import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ipd
from tqdm import tqdm
from image_helper import ImageHelper
import torch
import time


class VideoHelper:
    def __init__(self) -> None:
        self.image_helper = ImageHelper()

    def play_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    def play_video_in_notebook(self, video_path):
        ipd.display(ipd.Video(video_path))

    def get_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return frames

    def plot_video_frames(self, video_path):
        fig, axs = plt.subplots(4, 4, figsize=(30, 20))
        axs = axs.flatten()

        cap = cv2.VideoCapture(video_path)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        img_idx = 0
        for frame in range(n_frames):
            ret, img = cap.read()
            if not ret:
                break

            if frame % 100 == 0:
                axs[img_idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axs[img_idx].set_title(f'frame {frame}')
                axs[img_idx].axis('off')
                img_idx += 1

        plt.tight_layout()
        plt.show()
        cap.release()

    def downsample_video_and_save(self, video_path, output_path, downsample_factor=4):
        # Open the input video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        # Get video information (e.g., frame width, frame height, frames per second, etc.)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Downsampled frame dimensions
        downsampled_width = frame_width // downsample_factor
        downsampled_height = frame_height // downsample_factor

        print(
            f'orginal frame dimensions: frame_width = {frame_width}, frame_height = {frame_height}, fps = {fps}')
        print(
            f'downsampled frame dimensions: frame_width = {downsampled_width}, frame_height = {downsampled_height}, fps = {fps}')

        # Define the codec and create a VideoWriter object to save the downsampled frames
        # You can change the codec as needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # check if output_path is a directory ends with .mp4
        if output_path.endswith('.mp4'):
            output_filename = output_path
        else:
            # add .mp4 to the end of the output_path
            output_filename = output_path + '.mp4'

        # Create the VideoWriter object to save the frames
        out = cv2.VideoWriter(output_filename, fourcc, fps,
                              (downsampled_width, downsampled_height), isColor=True)

        # Get the total number of frames in the input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames,
                            desc='Processing Frames', unit='frames')

        index = 0

        # Loop through each frame in the video
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            index += 1

            # Downsample the frame by a the downsample_fatcor in both dimensions
            downsampled_frame = cv2.resize(
                frame, None, fx=1/downsample_factor, fy=1/downsample_factor, interpolation=cv2.INTER_AREA)

            if index == 1:
                print('downsampled_frame shape =', downsampled_frame.shape)
                self.image_helper.display_cv2_img(
                    downsampled_frame, show_grid=True)

            # # Write the downsampled frame to the output video
            out.write(downsampled_frame)

            # Update the progress bar
            progress_bar.update(1)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and output writer objects
        cap.release()
        out.release()

        # Close any open windows
        cv2.destroyAllWindows()

        # # Close the tqdm progress bar
        progress_bar.close()

        print(
            f"Video has been downsampled by a factor of {downsample_factor} and saved as", output_filename)

        return output_filename

    def apply_model_to_video_and_save(self, video_path, output_path, model, upsample_factor=4, should_upsample_first=False):
        if should_upsample_first:
            print("should_upsample_first must only be true for SRCNN!!!")

        if model is None:
            raise Exception('Error: Model is not defined.')

        # Open the input video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        # Get video information (e.g., frame width, frame height, frames per second, etc.)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # upsampled frame dimensions (factor of 4)
        upsampled_width = frame_width * upsample_factor
        upsampled_height = frame_height * upsample_factor

        print(
            f'orginal frame dimensions: frame_width = {frame_width}, frame_height = {frame_height}, fps = {fps}')
        print(
            f'upsampled frame dimensions: frame_width = {upsampled_width}, frame_height = {upsampled_height}, fps = {fps}')

        # Define the codec and create a VideoWriter object to save the downsampled frames
        # You can change the codec as needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # check if output_path is a directory ends with .mp4
        if output_path.endswith('.mp4'):
            output_filename = output_path
        else:
            # add .mp4 to the end of the output_path
            output_filename = output_path + '.mp4'

        # Create the VideoWriter object to save the frames
        out = cv2.VideoWriter(output_filename, fourcc, fps,
                              (upsampled_width, upsampled_height))

        # Get the total number of frames in the input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames,
                            desc='Processing Frames', unit='frames')

        index = 0

        # Loop through each frame in the video
        while True:
            with torch.no_grad():

                ret, frame = cap.read()

                if not ret:
                    break  # Break the loop if we have reached the end of the video

                index += 1

                frame_as_pil_img = self.image_helper.cv2_to_pil(frame)

                frame_after_model_tensor = self.image_helper.apply_model_to_image(
                    model.to('cpu'),
                    frame_as_pil_img,
                    downsample_factor=1,  # No downsampling, since the frame is already downsampled
                    # makes no difference for anything actually, since the downsample_factor is 1 (no downsampling)
                    should_upsample=should_upsample_first,
                    unsqueeze=True,
                    device='cpu'
                )

                # frame from pytorch tensor to cv2 image
                frame_after_model = frame_after_model_tensor.squeeze().permute(1, 2, 0).numpy()
                frame_after_model = cv2.cvtColor(
                    frame_after_model, cv2.COLOR_RGB2BGR)

                if index == 1:
                    print('original shape =', frame.shape)
                    self.image_helper.display_cv2_img(
                        frame_after_model, show_grid=True)
                    plt.show()
                    # frame_after_model = cv2.cvtColor(frame_after_model_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                    # frame_after_model = (frame_after_model * 255).astype(np.uint8)
                    print(type(frame_after_model))
                    # frame_after_model = frame_after_model.astype(np.uint8)
                    # frame_after_model = np.clip(frame_after_model, 0, 255).astype(np.uint8)
                    # self.image_helper.display_cv2_img(frame_after_model, show_grid=True)
                    plt.show()
                    print('after model shape =', frame_after_model.shape)

                frame_after_model = cv2.cvtColor(frame_after_model_tensor.squeeze(
                    0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                frame_after_model = (frame_after_model * 255).astype(np.uint8)

                # frame_after_model = cv2.cvtColor(frame_after_model_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR)
                # frame_after_model = np.clip(frame_after_model, 0, 255).astype(np.uint8)

                # Write the downsampled frame to the output video
                out.write(frame_after_model)

                # Update the progress bar
                progress_bar.update(1)

        # Release the video capture and output writer objects
        cap.release()
        out.release()

        # Close any open windows
        cv2.destroyAllWindows()

        # Close the tqdm progress bar
        progress_bar.close()

        print(
            f"Video has been downsampled by a factor of {upsample_factor} and saved as", output_filename)

    def check_if_video_is_valid(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        print(
            f'frame_width = {frame_width}, frame_height = {frame_height}, fps = {fps}')

        cap.release()

    def get_first_frame_of_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")

        ret, frame = cap.read()
        if not ret:
            raise Exception("Error: Could not read first frame of video file.")

        cap.release()

        return frame

    def downsample_video_and_save_and_apply_to_models_and_save(
            self,
            input_video_path: str,
            models: list,
            downsample_factor: int = 4,
            upsample_factor: int = 4,
    ):
        if models is None:
            raise Exception('Error: Models is not defined.')
        if len(models) == 0:
            raise Exception('Error: Models is empty.')
        if not isinstance(models, list):
            raise Exception('Error: Models is not a list.')
        if downsample_factor != upsample_factor:
            raise Exception('Error: downsample_factor != upsample_factor')

        folder_to_save = self._get_folder_name_from_path(input_video_path)

        input_video_name = self._get_video_name_from_path(input_video_path)

        downsampled_video_name = f"{folder_to_save}/{input_video_name}_downsampled_factor_of_{downsample_factor}"

        # downsample video and save
        downsampled_video_path = self.downsample_video_and_save(
            input_video_path, downsampled_video_name, downsample_factor)

        upsampled_video_by_interpolation_path = self.upsample_video_by_interpolation_and_save(
            downsampled_video_path, f"{folder_to_save}/{input_video_name}_upsampled_by_interpolation_factor_of_{upsample_factor}", upsample_factor)

        # apply models to video and save
        for model in models:
            print()
            print(f"Applying model {model.__class__.__name__} to video")

            # time how long it takes to apply model to video
            start_time = time.time()

            should_upsample_first = False  # useless variable, but I'm too lazy to remove it

            video_path_to_apply_model_to = downsampled_video_path

            if model.__class__.__name__ == 'SRCNN':
                # For SRCNN we apply the model to the upsampled video
                video_path_to_apply_model_to = upsampled_video_by_interpolation_path

            # print(f"should_upsample_first = {should_upsample_first}")
            # print(f"model.__class__.__name__ = {model.__class__.__name__}")
            # print(f"downsampled_video_path = {downsampled_video_path}")

            self.apply_model_to_video_and_save(
                video_path_to_apply_model_to, f"{folder_to_save}/{input_video_name}_upsampled_factor_of_{upsample_factor}_model_{model.__class__.__name__}", model, upsample_factor, should_upsample_first=should_upsample_first)

            end_time = time.time()

            print(
                f"Finished applying model {model.__class__.__name__} to video")
            print(f"Time elapsed: {end_time - start_time} seconds")

        print('Finished applying all models to video and saving the results')

    def _get_video_name_from_path(self, video_path):
        import os

        # Get the base filename without the directory path
        base_filename = os.path.basename(video_path)

        # Split the base filename on the '.' character to get the parts
        filename_parts = base_filename.split('.')

        # Extract the desired part (assuming it's always the first part)
        desired_substring = filename_parts[0]

        return desired_substring
    
    def _get_folder_name_from_path(self, path):
        import os

        # Get the directory path using os.path.dirname
        directory_path = os.path.dirname(path)

        return directory_path

    def upsample_video_by_interpolation_and_save(self,
                                                 video_path,
                                                 output_path,
                                                 upsample_factor=4):
        # Open the input video file
        cap = cv2.VideoCapture(video_path)

        # Check if the video file was opened successfully
        if not cap.isOpened():
            raise Exception("Error: Could not open video file.")
        
        # Get video information (e.g., frame width, frame height, frames per second, etc.)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # upsampled frame dimensions (factor of 4)
        upsampled_width = frame_width * upsample_factor
        upsampled_height = frame_height * upsample_factor

        print()
        print('******** [START] Upsampling video by interpolation ********')
        print(
            f'orginal frame dimensions: frame_width = {frame_width}, frame_height = {frame_height}, fps = {fps}')
        print(
            f'upsampled frame dimensions: frame_width = {upsampled_width}, frame_height = {upsampled_height}, fps = {fps}')
        
        # Define the codec and create a VideoWriter object to save the downsampled frames
        # You can change the codec as needed
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # check if output_path is a directory ends with .mp4
        if output_path.endswith('.mp4'):
            output_filename = output_path
        else:
            # add .mp4 to the end of the output_path
            output_filename = output_path + '.mp4'

        # Create the VideoWriter object to save the frames
        out = cv2.VideoWriter(output_filename, fourcc, fps,
                              (upsampled_width, upsampled_height))
        
        # Get the total number of frames in the input video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames,
                            desc='Processing Frames', unit='frames')
        
        index = 0

        # Loop through each frame in the video
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            index += 1

            # Upsample the frame by a the upsample_factor in both dimensions
            upsampled_frame = cv2.resize(
                frame, None, fx=upsample_factor, fy=upsample_factor, interpolation=cv2.INTER_CUBIC)
            
            if index == 1:
                print('original shape =', frame.shape)
                self.image_helper.display_cv2_img(
                    upsampled_frame, show_grid=True)
                plt.show()
                print('after interpolation shape =', upsampled_frame.shape)

            # Write the upsampled frame to the output video
            out.write(upsampled_frame)

            # Update the progress bar
            progress_bar.update(1)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and output writer objects
        cap.release()

        # Close any open windows
        cv2.destroyAllWindows()

        # Close the tqdm progress bar
        progress_bar.close()

        print(
            f"Video has been upsampled by a factor of {upsample_factor} and saved as", output_filename)
        
        print('******** [END] Upsampling video by interpolation ********')

        return output_filename



