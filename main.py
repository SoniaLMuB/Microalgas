import sys
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from sklearn.metrics import confusion_matrix

"""
CANTU SANCHEZ NUBIA ESMERALDA
MUÑOZ BARRIENTOS SONIA LIZBETH
GARCIA PUENTE LILIAN SAYLI
RODRIGUEZ MORENO JORGE JHOVAN
GUEVARA GARCIA JORGE
"""

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Detección de Microalgas'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        # Layout principal
        mainLayout = QVBoxLayout()

        # Label para mostrar la imagen
        self.imageLabel = QLabel(self)
        mainLayout.addWidget(self.imageLabel)

        # Label para mostrar la leyenda de colores
        colorLegend = """
        Neural Net: Verde
        QDA: Azul
        SGDClassifier: Rojo
        NearestCentroid: Amarillo
        Coincidencias entre clasificadores: Morado
        """
        self.colorLegendLabel = QLabel(colorLegend, self)
        mainLayout.addWidget(self.colorLegendLabel)

        # Layout horizontal para el botón
        hLayout = QHBoxLayout()

        # Botón para cargar la imagen
        self.loadImageButton = QPushButton('Cargar Imagen', self)
        self.loadImageButton.clicked.connect(self.loadImage)
        hLayout.addWidget(self.loadImageButton)

        # Agregar el layout horizontal al layout principal
        mainLayout.addLayout(hLayout)

        # Widget central
        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        self.setCentralWidget(centralWidget)

        self.names = ["Neural Net", "QDA", "SGDClassifier", "NearestCentroid"]

        self.classifiers = [
            MLPClassifier(alpha=1, max_iter=3000),
            QuadraticDiscriminantAnalysis(),
            SGDClassifier(max_iter=5),
            NearestCentroid(),
        ]
                # Añadir esto después de inicializar tus clasificadores y nombres
        self.combination_colors = {
            ("Neural Net", "QDA"): (0, 128, 0),  # Verde oscuro
            ("Neural Net", "SGDClassifier"): (128, 0, 128),  # Morado
            ("Neural Net", "NearestCentroid"): (0, 128, 128),  # Verde azulado oscuro
            ("QDA", "SGDClassifier"): (128, 128, 0),  # Amarillo oscuro
            ("QDA", "NearestCentroid"): (0, 0, 128),  # Azul oscuro
            ("SGDClassifier", "NearestCentroid"): (128, 0, 0)  # Rojo oscuro
        }

        self.show()


    def loadImage(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Cargar Imagen", "", "Imágenes (*.png *.jpg *.jpeg);;Todos los archivos (*)", options=options)
        if filePath:
            # Ejecutar el código de detección aquí
            processedImagePath = self.detectMicroalgas(filePath)

            # Mostrar la imagen con las detecciones
            pixmap = QPixmap(processedImagePath)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setAlignment(Qt.AlignCenter)

    def detectMicroalgas(self, filePath):
        # Aquí va todo el código de detección que proporcionaste
        original = cv2.imread(filePath)
        original2 = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        image = cv2.medianBlur(original2, 3)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        temporal = original2 * image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        temporal = cv2.morphologyEx(temporal, cv2.MORPH_OPEN, kernel, iterations=1)
        temporal = cv2.morphologyEx(temporal, cv2.MORPH_CLOSE, kernel, iterations=1)
        ret, image = cv2.threshold(temporal, 100, 250, cv2.THRESH_BINARY)

        cnts, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        ListaElementos = []
        for component in zip(cnts, hierarchy):
            currentContour = component[0]
            currentHierarchy = component[1]
            x, y, w, h = cv2.boundingRect(currentContour)
            if currentHierarchy[2] < 0:
                area = cv2.contourArea(currentContour)
                M = cv2.moments(currentContour)
                E = self.elongation(M)
                if area < 60:
                    roi = original[x:x+w, y:y+h, :]
                    ListaElementos.append([w, h, x, y])
                    cv2.drawContours(original, [currentContour], -1, (0, 0, 255), 2)

        ListaElementos2 = [i for i in ListaElementos]

        predictions_map = {}

        for name, clf in zip(self.names, self.classifiers):
            df = pd.read_csv('microalgas_dataset.csv')
            mymap = {'si': 1, 'no': 2}
            df = df.applymap(lambda s: mymap.get(s) if s in mymap else s)
            X = df.iloc[:, 0:128]
            y = list(df['class'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(X_train, y_train)
            Y_pred = clf.predict(X_test)
            predictions_map[name] = Y_pred
        
            cm = confusion_matrix(y_test,Y_pred)
            print (cm)

        def find_and_print_matches(predictions_map, ListaElementos2):
            matches = []
            matchTexts = []
            for index, (name1, preds1) in enumerate(predictions_map.items()):
                for name2, preds2 in list(predictions_map.items())[index + 1:]:
                    current_matches = np.logical_and(preds1 == 1, preds2 == 1)
                    for i, match in enumerate(current_matches):
                        if match and i < len(ListaElementos2):  # Asegurarse de que i es un índice válido
                            coords = ListaElementos2[i]
                            matchText = f"Coincidencia en coordenadas {coords[2]}, {coords[3]} (x,y) entre {name1} y {name2}\n"
                            matchTexts.append(matchText)
                            matches.append(((name1, name2), coords))
            
            # Guardar las coincidencias en un archivo .txt
            with open("matches.txt", "w") as f:
                f.writelines(matchTexts)
            
            return matches
        
        all_detections_image = original.copy()

        # Definir un mapa de colores para cada clasificador
        color_map = {
            "Neural Net": (0, 255, 0),      # Verde
            "QDA": (255, 0, 0),             # Azul
            "SGDClassifier": (0, 0, 255),   # Rojo
            "NearestCentroid": (0, 255, 255)  # Amarillo
        }

        # Primero, dibuja las detecciones individuales
        for name in self.names:
            Y_pred = predictions_map[name]
            self.draw_predictions(all_detections_image, Y_pred, ListaElementos2, color=color_map[name])

        # Luego, busca coincidencias y dibuja las coincidencias con prioridad
        matches = find_and_print_matches(predictions_map, ListaElementos2)
        for (name1, name2), coords in matches:
            color = self.combination_colors.get((name1, name2), (255, 255, 255))  # Usa blanco por defecto si no encuentra el color
            w, h, x, y = coords
            cv2.rectangle(all_detections_image, (x, y), (x + w, y + h), color, 2)

        # Redimensionar la imagen a las dimensiones deseadas (por ejemplo, 800x600)
        resized_image = cv2.resize(all_detections_image, (800, 600))

        # Guarda la imagen procesada en un archivo temporal
        tempFilePath = "temp_processed_image10_1.png"
        cv2.imwrite(tempFilePath, resized_image)

        return tempFilePath

    def elongation(self,m):
        x = m['mu20'] + m['mu02']
        y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
        if (x - y**0.5) == 0:
            denom = 1
        else:
            denom = (x - y**0.5)
        return (x + y**0.5) / denom

    def draw_predictions(self,image, predictions, coordinates, color=(0, 255, 0)):
        for label, (w, h, x, y) in zip(predictions, coordinates):
            if label == 1:
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        return image
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
