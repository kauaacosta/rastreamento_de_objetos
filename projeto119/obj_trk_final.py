import cv2

# Função para desenhar o retângulo e adicionar o texto
def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
    cv2.putText(img, "Rastreando", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

# Carregar o classificador pré-treinado de corpo
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Captura de vídeo
cap = cv2.VideoCapture("C:/Users/comma/OneDrive/Área de Trabalho/Programação/projeto119/footvolleyball.mp4")

# Inicializar o rastreador
tracker = cv2.TrackerMIL_create()

# Obter o primeiro quadro para rastrear
ret, frame = cap.read()
bbox = cv2.selectROI("Rastreamento de Objeto", frame, False)
tracker.init(frame, bbox)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Rastrear o objeto no quadro atual
    success, bbox = tracker.update(frame)

    if success:
        # Desenhar retângulo e adicionar texto se o rastreamento for bem-sucedido
        drawBox(frame, bbox)
    else:
        # Exibir "Errou" na tela se o rastreamento falhar
        cv2.putText(frame, "Errou", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Rastreamento de Objeto', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Pressione 'Esc' para sair
        break

cap.release()
cv2.destroyAllWindows()