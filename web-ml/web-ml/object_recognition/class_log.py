from csv import DictWriter


class Log:
    def __init__(self, file):
        self.log_writer = DictWriter(
            file,
            fieldnames=[
                'time',
                'class_name',
                'class_confidence',
                'plate',
                'plate_confidence',
                'base64'
            ]
        )
        self.log_writer.writeheader()

    def add_record(self, azs, trk, time, class_name, class_confidence, plate, plate_confidence, base64):
        self.log_writer.writerow(
            {
                'time': time,
                'class_name': class_name,
                'class_confidence': class_confidence,
                'plate': plate,
                'plate_confidence': plate_confidence,
                'base64': base64
            }
        )
