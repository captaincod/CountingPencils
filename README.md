# Определение количества карандашей на наборе изображений
Вход: набор изображений из папки images/  
Выход: суммарное количество карандашей на всех изображениях  
Во время выполнения программа показывает обрабатываемое изображение и, если на нём есть карандаш, рисует его контур  
Обнаруженные карандаши и их свойства можно посмотреть в log.txt  
В log можно увидеть, как отличаются высота и ширина (h и w) карандашей на изображениях, где они расположены под наклоном, от тех, где они расположены вертикально/горизонтально.  
Поэтому если определять, является ли объект карандашом или нет, по h и w, то повернутые карандаши не пройдут проверку. Для решения этого используется cv2.boxPoints. 
