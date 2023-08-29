# RSNA/KCR Abstract

- **전처리 과정**은 신체의 여러 단면으로 이루어진 CT 사진을 3D로 재구성하고, 3D volume을 뼈가 잘 보이는 각도로 회전시켜 실제 X-Ray 사진과 비슷한 효용성을 가지는 가짜 X-Ray(DRR) 사진을 생성하는 것이다. (총 131장)
- DRR과 해당하는 mask를 데이터셋으로 사용해서 어떻게 적은 수의 데이터를 학습에 활용하여 그 representation을 학습할지 실험한다.

- CT 영상을 DRR(Digitally Reconstructed Radiograph)로 전처리하여 만들어진 매우 적은 데이터로, 겹쳐 있는 손목 (wrist) 뼈를 semantic segmentation (제 1저자로 submission)
  * drr_wrist_abstract.pdf (Graphic Abstract)
  * drr_wrist.pdf (RSNA Abstract)

- CT 영상을 DRR(Digitally Reconstructed Radiograph)로 전처리하여 만들어진 매우 적은 데이터로, 하체의 근육/지방 segmentation (공동 1저자로 submission)
  * drr_lower_extremity.pdf (RSNA Abstract)
  * drr_lower_extremity_abstract.pdf (Graphic Abstract)

