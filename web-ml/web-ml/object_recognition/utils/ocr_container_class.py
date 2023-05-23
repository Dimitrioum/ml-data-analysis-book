from .config import MASK_RCNN_DIR, MASK_RCNN_LOG_DIR
# from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing


class ContainerOCR:
    def __init__(self):
        pass
        # Initialize npdetector with default configuration file.
        # self.nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
        # self.nnet.loadModel("latest")

        # self.rectDetector = RectDetector()
        #
        # self.optionsDetector = OptionsDetector()
        # self.optionsDetector.load("latest")
        #
        # # Initialize text detector.
        # self.textDetector = TextDetector.get_static_module("eu")()
        # self.textDetector.load("latest")

    def cast(self, image: 'ndarray') -> 'list[str]':
        # Detect points. TODO batch
        try:
            arrPoints = self.rectDetector.detect(
                filters.cv_img_mask(
                    self.nnet.detect([image])
                )
            )
            if arrPoints.size == 0:
                return []
            zones = self.rectDetector.get_cv_zonesBGR(image, arrPoints)
            regionIds, stateIds, countLines = self.optionsDetector.predict(zones)
            regionNames = self.optionsDetector.getRegionLabels(regionIds)

            # find text with postprocessing by standart
            textArr = self.textDetector.predict(zones)
            textArr = textPostprocessing(textArr, regionNames)
            return textArr
        except IndexError:
            print('bad frame')
            return None


    def forward(self, rois: list):
        """ Recognition of car license plates
        Args:
            rois: list of rois from the original video frame
        Return:
            plate: string with recognized characters
            proba: corresponding probabilities of character recognition
        """
        licence_classifier = general_model.TRTLicenceClass(self.licence_config, fp16_mode=True)
        for roi in rois:
            model_output = licence_classifier(roi)


        return plate, proba
