#IMPORTS
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


#Inicializando o estado inicial de cada braço
left_up = False
right_up = False
left_down = False
right_down = False

#Inicializando ângulos
angle_elbow_left = 0
angle_elbow_right = 0

#Contador de reps
counter = 0

#Capturando vídeo
cap = cv2.VideoCapture(0)

#Cálculo dos ângulos
def calculate_angle(a, b, c):
  a = np.array(a) #Primeiro
  b = np.array(b) #Meio
  c = np.array(c) #Final

  radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
  angle = np.abs(radians*180.0/np.pi)

  if angle > 180.0:
    angle = 360 - angle

  return angle

#Configurando instância do medipipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        #Recolorindo a imagem para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #Fazendo as detecções
        results = pose.process(image)

        #Recolorindo para BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Obtendo o tamanho da imagem
        image_height, image_width, _ = image.shape

        #Extraindo os pontos de referência
        try:
            landmarks = results.pose_landmarks.landmark

            #Pegando coordenadas 

            # lado esquerdo
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # lado direito 
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            #Calculando ângulos

            # lado esquerdo
            angle_elbow_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_shoulder_left = calculate_angle(hip_left, shoulder_left, elbow_left)

            # lado direito
            angle_elbow_right  = calculate_angle(shoulder_right, elbow_right, wrist_right)
            angle_shoulder_right = calculate_angle(hip_right, shoulder_right, elbow_right)

            #Visualizando ângulos

            # lado esquerdo
            cv2.putText(image, f'{angle_elbow_left:.2f}',
                        tuple(np.multiply(elbow_left, [image_width,image_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f'{angle_shoulder_left:.2f}',
                        tuple(np.multiply(shoulder_left, [image_width,image_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )

            # lado direito
            cv2.putText(image, f'{angle_elbow_right:.2f}',
                        tuple(np.multiply(elbow_right, [image_width,image_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )
            cv2.putText(image, f'{angle_shoulder_right:.2f}',
                        tuple(np.multiply(shoulder_right, [image_width,image_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA
                        )

            #Lógica para contagem de repetições rosca biceos

            if angle_shoulder_left < 30 and 150 <= angle_elbow_left <= 180:
                left_down = True

            elif angle_shoulder_right < 30 and 150 <= angle_elbow_right <= 180:
                right_down = True

            if left_down and angle_shoulder_left < 30 and angle_elbow_left < 70:
                left_up = True
                left_down = False
            
            elif right_down and angle_shoulder_right < 30 and angle_elbow_right < 70:
                right_up = True
                right_down = False

            # Se um dos braços fez o movimento
            if left_up or right_up:
                counter += 1
                print(counter)

                # Resetar o estado do braço que fez o movimento
                left_up = False
                right_up = False

        except:
            pass

        #Status
        cv2.rectangle(image, (0,0), (125,73), (0,0,0), -1)

        #Barra de Progresso
        average_angle = (angle_elbow_left + angle_elbow_right) / 2
        bar_val = np.interp(average_angle,(40,155),(60,300+60))
        cv2.rectangle(image,(560,int(bar_val)),(40+560, 300+60),(0,255,0), cv2.FILLED)
        cv2.rectangle(image, (560, 60), (40+560,300+60), (0,0,0),2)

        #Contagem
        cv2.putText(image, 'REPS', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        #Renderizando as detecções
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(57, 244, 1), thickness=4, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(12, 192, 9), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()