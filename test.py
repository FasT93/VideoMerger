from vidmergerImagination import VideoMerger

path = 'E:/Databases/EEG/Video Observing Task/VideoImagination/Stimuli/!Sborki/V9_Dima/_Session2'

vm = VideoMerger(pause_duration = 12, pause_delta=0, transitionDuration=2)
vm.merge(path, depth=None)