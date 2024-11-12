import websockets
import asyncio
import cv2
import base64
import time
from collections import deque

# WebSocket parameters
port = 5000
host = '127.0.0.1'

# Video capture parameters
video_file_path = "C:/areffile/anprv7/engine/output/rooz.mp4"  # Replace with the path to your video
cap = cv2.VideoCapture(video_file_path)

FRAME_RATE = 30  # Set target frame rate
SCALE_FACTOR = 0.5  # Resolution downscale factor
JPEG_QUALITY = 30  # Lower quality for faster encoding

# Frame buffer for asynchronous handling
frame_buffer = deque(maxlen=5)  # Buffer to store recent frames

print(f"Started WebSocket server on ws://{host}:{port}")

async def capture_frames():
    """Capture frames from the video file and store them in the buffer asynchronously."""
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video playback finished.")
            break

        # Resize and encode the frame
        frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        data = base64.b64encode(encoded).decode('utf-8')
        
        # Store frame in the buffer
        if len(frame_buffer) < frame_buffer.maxlen:
            frame_buffer.append(data)

        # Wait based on desired FPS
        await asyncio.sleep(1 / FRAME_RATE)

async def transmit(websocket, path=None):
    """Transmit frames from the buffer to connected WebSocket clients."""
    print("Client Connected!")
    try:
        while True:
            if frame_buffer:
                # Get the latest frame from the buffer
                data = frame_buffer.popleft()
                
                # Send the frame over WebSocket
                await websocket.send(data)
            await asyncio.sleep(0.01)  # Small delay to prevent busy loop

    except websockets.ConnectionClosed:
        print("Client Disconnected!")

async def main():
    """Main function to start the WebSocket server and frame capture."""
    server = await websockets.serve(transmit, host, port)
    print("WebSocket server started.")
    await capture_frames()  # Start frame capture
    await server.wait_closed()  # Keep the server running

# Use asyncio.run() to start the application
if __name__ == "__main__":
    asyncio.run(main())

cap.release()
