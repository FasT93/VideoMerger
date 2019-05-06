import cv2
import os
import numpy as np
import random as rnd

from pymediainfo import MediaInfo

class VideoMerger():
    def __init__(self, output_format='avi', size=None, shuffle=True,
                 marker=True, marker_size=0.08, marker_offset=0.05, marker_duration=1,
                 pause_duration=3, pause_delta=0, last_pause=True,
                 ready_duration=5, thanks_duration=5, fixed_classes=None):
        self.vids = []
        self.basenames = []
        self.depth = None
        self.__formats = ('.avi', '.mp4')

        if output_format == 'avi':
            self.__fourcc = cv2.VideoWriter_fourcc(*'XVID')
        elif output_format == 'mp4':
            self.__fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        else:
            print("Unknown output video format: '{0}'! Using .avi format!".format(output_format))
            self.__fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_format = 'avi'
        self.output_format = output_format

        self.size = size
        self.shuffle = shuffle
        self.marker = marker
        self.marker_size = marker_size
        self.marker_offset = marker_offset
        self.marker_duration = marker_duration
        self.pause_duration = pause_duration
        self.pause_delta = pause_delta
        self.last_pause = last_pause
        self.ready_duration = ready_duration
        self.thanks_duration = thanks_duration
        self.fixed_classes = fixed_classes

    def merge(self, input_folder, output_folder='newvideo', depth=None, show=False):
        self.vids.clear()
        self.depth = depth
        if isinstance(input_folder, list):
            for inp in input_folder:
                if inp[-1] == '/' or inp[-1] == '\\':
                    inp = inp[:-1]
                self.__getAllVideos(inp)
        else:
            if input_folder[-1] == '/' or input_folder[-1] == '\\':
                input_folder = input_folder[:-1]
            self.__getAllVideos(input_folder)

        assert len(self.vids) > 0, 'No video data!'

        if self.fixed_classes is None:
            self.basenames = sorted(list(set(self.basenames)))
        else:
            self.basenames = self.fixed_classes

        if self.shuffle:
            rnd.shuffle(self.vids)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        info = self.__getAllInfo()
        max_width = max([x['width'] for x in info])
        max_height = max([x['height'] for x in info])
        frame_rate = int(round(max([float(x['fps']) for x in info])))

        onsets = [0.0]
        textfont = cv2.FONT_HERSHEY_COMPLEX
        font_scale_ready = self.__getTextScale('Ready', 0.5 * max_width, 0.3 * max_height, textfont, -1)
        font_scale_thanks = self.__getTextScale('Thanks for your attention', 0.5 * max_width, 0.3 * max_height, textfont, -1)
        out = cv2.VideoWriter(output_folder + '/video.' + self.output_format, self.__fourcc, frame_rate, (max_width, max_height))
        black = np.zeros((max_height, max_width, 3), np.uint8)

        # Ready screen
        for t in range(int(frame_rate * self.ready_duration)):
            out.write(self.__drawText(black, 'Ready', textfont, font_scale=font_scale_ready))

        # Go throw all videos
        for i, vid in enumerate(self.vids):
            print('Processing {0}/{1}'.format(i + 1, len(self.vids)))

            marker_counter = 0

            # Saving video
            cap = cv2.VideoCapture(vid)
            video_duration = 0
            while (cap.isOpened()):
                ret, frame = cap.read()
                if ret:
                    # Resizing image
                    if max_width != frame.shape[1] or max_height != frame.shape[0]:
                        frame = self.__resizeImage(frame, (max_width, max_height))

                    # Adding white marker
                    if self.marker:
                        if self.marker_duration > 0 and marker_counter / frame_rate <= self.marker_duration:
                            frame = self.__drawMarker(frame, self.marker_size, self.marker_offset)
                            marker_counter += 1
                        else:
                            frame = self.__drawMarker(frame, self.marker_size, self.marker_offset, color=(0, 0, 0))

                    # Showing videos
                    if show:
                        cv2.imshow('Video', frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break

                    # Write to videofile
                    out.write(frame)
                    video_duration += 1
                else:
                    break
            cap.release()

            # Adding pause
            current_pause_duration = (2 * rnd.random() - 1) * self.pause_delta + self.pause_duration
            if (self.last_pause is True) or (self.last_pause is False and i + 1 < len(self.vids)):
                for t in range(int(frame_rate * current_pause_duration)):
                    if t / frame_rate <= self.marker_duration:
                        out.write(self.__drawMarker(black, self.marker_size, self.marker_offset))
                    else:
                        out.write(black)

            # Saving duration
            onsets.append(video_duration / frame_rate + current_pause_duration)

        # Thanks for attention screen
        for t in range(int(frame_rate * self.thanks_duration)):
            out.write(self.__drawText(black, 'Thanks for your attention', textfont, font_scale=font_scale_thanks))

        out.release()
        cv2.destroyAllWindows()

        self.__writeToFile(output_folder + '/exemplars.txt', self.vids)
        self.__writeToFile(output_folder + '/onsets.txt', [str(x) for x in np.cumsum(onsets[:-1])])
        self.__writeToFile(output_folder + '/labels.txt', [str(self.basenames.index(os.path.basename(os.path.dirname(v)))) for v in self.vids])

    def __resizeImage(self, image, size):
        w, h = image.shape[1], image.shape[0]
        aspect_ratio = w / h
        w_ratio = size[0] / w
        h_ratio = size[1] / h
        if w_ratio < h_ratio:
            new_w = size[0]
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = size[1]
            new_w = int(new_h * aspect_ratio)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        x_offset = (size[0] - new_w) // 2
        y_offset = (size[1] - new_h) // 2

        if len(image.shape) > 2:
            result = np.zeros((size[1], size[0], image.shape[2]), np.uint8)
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w, :] = resized
        else:
            result = np.zeros((size[1], size[0]), np.uint8)
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return result

    def __getAllVideos(self, path, curdepth=0):
        if self.depth is not None:
            if curdepth >= self.depth:
                return

        if os.path.isdir(path):
            for file in os.listdir(path):
                fullpath = path + '/' + file
                if os.path.isdir(fullpath):
                    self.__getAllVideos(fullpath, curdepth + 1)
                elif os.path.isfile(fullpath):
                    if file.endswith(self.__formats):
                        self.vids.append(fullpath)
                        dirname = os.path.dirname(fullpath)
                        self.basenames.append(os.path.basename(dirname))
        elif os.path.isfile(path):
            if path.endswith(self.__formats):
                self.vids.append(path)
                dirname = os.path.dirname(path)
                self.basenames.append(os.path.basename(dirname))

    def __getFileDetails(self, filepath):
        media_info = MediaInfo.parse(filepath)
        result = {}
        for track in media_info.tracks:
            if track.track_type == 'Video':
                result['fps'] = track.frame_rate
                result['width'] = track.width
                result['height'] = track.height
        return result

    def __getAllInfo(self):
        info = []
        for vid in self.vids:
            info.append(self.__getFileDetails(vid))
        return info

    # size - proportion of minimal size
    # offset - proportion of minimal size
    def __drawMarker(self, img, size=0.08, offset=0.05, color=(255, 255, 255)):
        temp = img.copy()
        w, h = temp.shape[1], temp.shape[0]
        minsize = min((w, h))
        r = int((minsize * size) / 2)
        offset = int(minsize * offset)
        cv2.circle(temp, (offset + r, h - offset - r), r, color, thickness=-1, lineType=cv2.LINE_AA)
        return temp

    def __getTextScale(self, text, target_width, target_height, font, thickness, s_start=0.5, s_end=20, s_points=50):
        s_prev = 0
        for s in np.linspace(s_start, s_end, s_points):
            (w, h), _ = cv2.getTextSize(text, font, s, thickness)
            if h > target_height or w > target_width:
                return s_prev
            s_prev = s
        return s_end

    def __drawText(self, img, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, color=(255, 255, 255)):
        temp = img.copy()
        (w, h), _ = cv2.getTextSize(text, font, font_scale, -1)
        pos = (int((img.shape[1] - w) / 2), int((img.shape[0] - h) / 2) + h)
        cv2.putText(temp, text, pos, font, font_scale, color, lineType=cv2.LINE_AA)
        return temp

    def __writeToFile(self, filename, data):
        with open(filename, 'w') as f:
            for d in data:
                f.write(d + '\n')