"""
웹캠 잘 연결됐는지 확인하는 코드
"""
import cv2

# # 웹캠 열기 (기본 장치는 index 0)
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("❌ 웹캠을 열 수 없습니다.")
# else:
#     print("✅ 웹캠이 정상적으로 열렸습니다. 창을 닫으려면 'q'를 누르세요.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("❌ 프레임을 읽을 수 없습니다.")
#             break

#         # 화면에 프레임 보여주기
#         cv2.imshow('Webcam Test', frame)

#         # 'q' 키 누르면 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# 0, 1, 2, 3... 순차적으로 시도해 보기
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ 카메라 {i}번 열림!")
        break
    else:
        print(f"❌ 카메라 {i}번 없음.")
    cap.release()
