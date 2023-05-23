
import argparse
import cv2


def main():
    """
    Утилита для обнаружения координат регионов интереса
    для записи в config.py новых объектов.
    """
    parser = argparse.ArgumentParser(description='Get dot coordinates.')
    parser.add_argument('img_address', action = 'store', type=str, help = 'Input: img_address', default=None)
    args = parser.parse_args()

    address = args.img_address

    refPt = []
    cropping = False

    def click_and_crop(event, x, y, flags, param):
        global refPt, cropping

        if event == cv2.EVENT_LBUTTONDOWN:
            refPt = [(x, y)]
            cropping = True
        elif event == cv2.EVENT_LBUTTONUP:

            refPt.append((x, y))
            cropping = False

            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
            print(f'Coordinates: {refPt[0]}')

    image = cv2.imread(address)
    clone = image.copy()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", click_and_crop)

    while True:
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        if len(refPt) == 2:
            roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        if key == ord("r"):
            image = clone.copy()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()