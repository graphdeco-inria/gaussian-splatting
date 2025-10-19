::CapturingReality

:: switch off console output
::@echo off
@echo on
set RootFolder=%1

:: path to RealityCapture application
set RealityCaptureExe="C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe"

:: variable storing path to images for texturing model
set Project="%RootFolder%\rcProj\RCproject.rcproj"

:: run RealityCapture
:: test and fix video import when RC working again

%RealityCaptureExe% -load %Project% ^
        -selectAllImages ^
        -enableAlignment false ^
        -selectImage *test_* ^
        -enableAlignment true ^
       
        




