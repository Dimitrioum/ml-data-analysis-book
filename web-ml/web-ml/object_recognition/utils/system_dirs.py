import os


class SystemDir(object):

    def __init__(self,
                 dirs=['dataset', 'predictions', 'rgbplates',
                       'rgb_roi', 'grayscale_roi'], check=True):
        self.check = check
        self.dirs = dirs

    def check_dirs(self):
        main_wd = os.getcwd().split(os.sep + 'utils')[0]
        files = os.listdir(main_wd)

        for i, folder in enumerate(self.dirs):
            if folder in files:
                pass
            else:
                os.mkdir(os.path.join(main_wd, folder))
                print('|{}|SystemDir|: Created {}'.format(i,
                                                          os.path.join(main_wd,
                                                                       folder)))
