import cv2
from scipy.spatial import distance

log = open("log.txt", "w")
count = 0

for img in range(1, 13):
    raw = cv2.imread(f"images/img ({img}).jpg")
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (7, 7), 0)
    _, thresh = cv2.threshold(image, 120, 255, 0)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 10)

    pencil_number = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        points = cv2.boxPoints(cv2.minAreaRect(cnt))
        w_euc = distance.euclidean(points[0], points[1])
        h_euc = distance.euclidean(points[0], points[3])
        if (h_euc > 3 * w_euc and h_euc > 1000) or (w_euc > 3 * h_euc and w_euc > 1000):
            log.write(f"picture: {img}, pencil: {pencil_number}, x: {x}, y: {y}, w: {w}, h: {h}\n")
            pencil_number += 1
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 20)
            count += 1
    cv2.namedWindow(f"Image {img}", cv2.WINDOW_KEEPRATIO)
    cv2.imshow(f"Image {img}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

log.close()
print("Суммарное количество карандашей на изображениях:", count)

# out: Суммарное количество карандашей на изображениях: 21
