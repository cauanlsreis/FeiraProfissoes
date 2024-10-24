# IMPORTS
import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Contador de reps
counter = 0
check = True

# Capturando vídeo
cap = cv2.VideoCapture(0)

# Configurando instância do MediaPipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolorindo a imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Fazendo as detecções
        results = pose.process(image)

        # Recolorindo para BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Obtendo o tamanho da imagem
        image_height, image_width, _ = image.shape

        # Extraindo os pontos de referência
        try:
            landmarks = results.pose_landmarks.landmark

            # Pegando coordenadas para as mãos e os pés

            # Lado esquerdo
            hand_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y * image_height
            ]
            foot_left = [
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * image_height
            ]

            # Lado direito 
            hand_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y * image_height
            ]
            foot_right = [
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * image_width,
                landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * image_height
            ]

            # Calculando distâncias
            distanciaMaos = math.hypot(hand_right[0] - hand_left[0], hand_right[1] - hand_left[1])
            distanciaPes = math.hypot(foot_right[0] - foot_left[0], foot_right[1] - foot_left[1])

            print(f'Mãos: {distanciaMaos:.2f}, Pés: {distanciaPes:.2f}')

            # Lógica para contagem de repetições (ajustar valores com base na escala da imagem)
            if check and distanciaMaos <= 150 and distanciaPes >= 150:
                counter += 1
                check = False
            if distanciaMaos >= 150 and distanciaPes <= 150:
                check = True
            
            # Mapear a distância para a barra de progresso (ajuste os valores conforme necessário)
            # Supondo que a distância mínima seja 40 e a máxima seja 200
            bar_val = np.interp(distanciaMaos, (40, 200), (60, 300 + 60))

            # Desenhar a barra de progresso para a distância das mãos
            cv2.rectangle(image, (560, int(bar_val)), (40 + 560, 300 + 60), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(image, (560, 60), (40 + 560, 300 + 60), (0, 0, 0), 2)

        except Exception as e:
            print(f"Erro ao processar landmarks: {e}")
            pass

        # Status da contagem
        cv2.rectangle(image, (0, 0), (125, 73), (0, 0, 0), -1)

        # Contagem
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Renderizando as detecções
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(57, 244, 1), thickness=4, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(12, 192, 9), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
