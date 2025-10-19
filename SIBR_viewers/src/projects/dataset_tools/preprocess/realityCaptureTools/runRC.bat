::CapturingReality

:: switch off console output
::@echo off
@echo on
set RootFolder=%1
set Video="%RootFolder%\videos\video.mp4"
set FPS=%2

set ConfigFolder=D:\Users\gdrett\src\sibr_core\install\scripts
:: path to RealityCapture application
set RealityCaptureExe="C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe"

:: variable storing path to images for creating model
set Images="%RootFolder%\images"
set TestImages="%RootFolder%\test"
set TrainImages="%RootFolder%\train"
set PathImages="%RootFolder%\train"

:: set a new name for calculated model
set ModelName="RCTest"

:: set the path, where model is going to be saved, and its name
set ModelObj="%RootFolder%\rcScene\meshes\mesh.obj"
set ModelXYZ="%RootFolder%\rcScene\meshes\point_cloud.xyz"

:: variable storing path to images for texturing model
set Project="%RootFolder%\rcproj\mesh.rcproj"

:: run RealityCapture
:: test and fix video import when RC working again

echo %@Images%

%RealityCaptureExe% -addFolder %TrainImages% ^
        -addFolder %TestImages% ^
        -importVideo %Video% %RootFolder%\video_frames\ %FPS% ^
        -align ^
        -selectMaximalComponent ^
        -selectAllImages ^
        -enableAlignment false ^
        -selectImage *test_* ^
        -enableAlignment true ^
        -exportRegistration %RootFolder%\rcScene\test_cameras\bundle.out %ConfigFolder%\registrationConfig.xml ^
        -selectAllImages ^
        -enableAlignment false ^
        -selectImage *frame* ^
        -enableAlignment true ^
        -exportRegistration %RootFolder%\rcScene\path_cameras\bundle.out %ConfigFolder%\registrationConfig.xml ^
        -selectAllImages ^
        -enableAlignment false ^
        -selectImage *train_* ^
        -enableAlignment true ^
        -exportRegistration %RootFolder%\rcScene\cameras\bundle.out %ConfigFolder%\registrationConfig.xml ^
        -setReconstructionRegionAuto ^
        -scaleReconstructionRegion 1.4 1.4 2.5 center factor ^
        -calculateNormalModel ^
        -selectMarginalTriangles ^
        -removeSelectedTriangles ^
        -calculateTexture ^
        -save %Project% ^
        -renameSelectedModel %ModelName% ^
        -exportModel %ModelName% %ModelObj% ^
        -exportModel %ModelName% %ModelXYZ% ^
        -quit
       
        




