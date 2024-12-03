from cx_Freeze import setup, Executable

setup(
    name="MyProgram",
    version="1.0",
    description="My Python Program",
    executables=[Executable("rtsp_streaming.py")],
)