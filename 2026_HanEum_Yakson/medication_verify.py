import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # 양손 감지
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

prev_time = 0
hand_near_mouth_start = None
hand_y_history = []
throat_history = []

def get_distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2) ** 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_result = hands.process(rgb)
    face_result = face_mesh.process(rgb)
    pose_result = pose.process(rgb)

    mouth_x, mouth_y = None, None
    hand_x, hand_y = None, None
    nose_y = None
    shoulder_y = None

    confidence = 0.0
    reasons = []

    upper_points = [13, 312, 311, 310]
    lower_points = [14, 317, 402, 318]

    # ---- Face Mesh ----
    if face_result.multi_face_landmarks:
        for face_landmarks in face_result.multi_face_landmarks:
            mp_draw.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

            mouth_x = int(sum(face_landmarks.landmark[i].x for i in upper_points) / len(upper_points) * w)
            mouth_y = int(sum(face_landmarks.landmark[i].y for i in upper_points) / len(upper_points) * h)

            upper_y = sum(face_landmarks.landmark[i].y for i in upper_points) / len(upper_points)
            lower_y = sum(face_landmarks.landmark[i].y for i in lower_points) / len(lower_points)
            lip_distance = (lower_y - upper_y) * h

            # 입 열림 임계값 낮춤 (5 → 2)
            if lip_distance > 4:
                confidence += 0.2
                reasons.append("Mouth Open")

            nose_y = face_landmarks.landmark[1].y * h

            throat_y = face_landmarks.landmark[152].y * h
            throat_history.append(throat_y)
            if len(throat_history) > 10:
                throat_history.pop(0)

            if len(throat_history) == 10:
                throat_movement = max(throat_history) - min(throat_history)
                if throat_movement > 8:
                    confidence += 0.2
                    reasons.append("Throat Movement")

    # ---- Hands (입에 가까운 손 선택) ----
    if hand_result.multi_hand_landmarks and mouth_x:
        min_distance = float('inf')
        closest_hand = None

        for hand_landmarks in hand_result.multi_hand_landmarks:
            hx = int(hand_landmarks.landmark[9].x * w)
            hy = int(hand_landmarks.landmark[9].y * h)
            dist = get_distance(hx, hy, mouth_x, mouth_y)

            if dist < min_distance:
                min_distance = dist
                closest_hand = hand_landmarks
                hand_x, hand_y = hx, hy

        # 선택된 손만 그리기
        if closest_hand:
            mp_draw.draw_landmarks(
                frame, closest_hand, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=3))

            # 손 높이 변화 추적
            hand_y_history.append(hand_y)
            if len(hand_y_history) > 15:
                hand_y_history.pop(0)

            if len(hand_y_history) == 15:
                hand_moved_up = hand_y_history[0] - hand_y_history[-1] > 10
                if hand_moved_up:
                    confidence += 0.15
                    reasons.append("Hand Moving Up")

    # ---- 손↔입 거리 ----
    if mouth_x and hand_x:
        distance = get_distance(hand_x, hand_y, mouth_x, mouth_y)

        if distance < 150:
            confidence += 0.3
            reasons.append("Hand Near Mouth")

            if hand_near_mouth_start is None:
                hand_near_mouth_start = time.time()
            else:
                elapsed = time.time() - hand_near_mouth_start
                if elapsed > 2.0:
                    confidence += 0.15
                    reasons.append(f"Hand Stayed {int(elapsed)}s")
        else:
            hand_near_mouth_start = None

        color_line = (0, 255, 0) if distance < 150 else (0, 200, 255)
        cv2.line(frame, (hand_x, hand_y), (mouth_x, mouth_y), color_line, 2)
        cv2.circle(frame, (mouth_x, mouth_y), 8, (0, 0, 255), -1)
        cv2.circle(frame, (hand_x, hand_y), 8, (255, 0, 0), -1)

    # ---- Pose ----
    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

        left_shoulder = pose_result.pose_landmarks.landmark[11]
        right_shoulder = pose_result.pose_landmarks.landmark[12]
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * h

        if nose_y and shoulder_y:
            if nose_y < shoulder_y * 0.6:
                confidence += 0.15
                reasons.append("Head Tilted")

    # ---- 최종 판정 ----
    confidence = min(confidence, 1.0)

    if confidence >= 0.8:
        verdict = "MEDICATION TAKEN"
        verdict_color = (0, 255, 0)
    elif confidence >= 0.4:
        verdict = "LIKELY TAKEN"
        verdict_color = (0, 200, 255)
    else:
        verdict = "WAITING..."
        verdict_color = (255, 255, 255)

    # ---- UI 패널 ----
    cv2.rectangle(frame, (0, 0), (400, 230), (0, 0, 0), -1)
    cv2.putText(frame, verdict, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, verdict_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, verdict_color, 2)

    for i, reason in enumerate(reasons):
        cv2.putText(frame, f"+ {reason}", (10, 95 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("YAKSON - Medication Verify", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
