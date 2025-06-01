# Qt 관련 모든 환경 변수를 비활성화
export QT_QPA_PLATFORM=offscreen
unset QTDIR
unset QT_PLUGIN_PATH
unset LD_LIBRARY_PATH

# KIME 입력기 관련 환경 변수도 비활성화
unset GTK_IM_MODULE
unset QT_IM_MODULE
unset XMODIFIERS

for scene in /home/cvnar/disk4tb/tandt/*; do
    scene_name=$(basename $scene)
    python convert.py -s /home/cvnar/disk4tb/tandt/$scene_name
done