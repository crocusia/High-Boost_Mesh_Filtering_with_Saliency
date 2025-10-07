⚠️ 본 저장소에는 지도 교수님께서 제공하신 3D Mesh 관련 코드가 비공개로 유지되어,<br> 
연구에서 사용된 전체 코드가 포함되어 있지 않습니다. <br>
이로 인해 본 저장소의 코드만으로는 프로젝트 전체를 실행할 수 없고,<br>
GPU 연산 및 관련 서브 모듈만 포함되어 있음을 알려드립니다. <br>
<hr>

### 비고
본 연구는 **`3차원 삼각형 메쉬의 과장을 안정적으로 표현할 수 있는 필터링과 GPU 최적화`** 연구의 후속 연구입니다. <br>
🔗[High-BoostMeshFiltering_GPU](https://github.com/crocusia/High-BoostMeshFiltering_GPU) <br>

# 🗃️ Summary 
High-Boost Mesh Filtering에서 새로운 법선 벡터를 계산하는 방법으로 인해 **경계 영역**에서 계단현상이 발생한다. <br>
Mesh Saliency를 통해 **지역적 중요도가 높은 영역(경계 영역)을 판별**하고, 강화를 위한 법선벡터 연산에 **Saliency 변화량 기반 가중치**를 적용한다.<br>
<br>

<img width="1799" height="379" alt="image" src="https://github.com/user-attachments/assets/a06c4772-513a-4453-8aeb-375b4d445120" />

<br>

- 경계 영역의 곡률을 유지를 통해 계단현상을 완화하며 안정적인 형상 강화
- 단순 곡률만 고려한 기존 방법에서는 누락된 Mesh의 중요도가 높은 특징 강화

# 연구 목표
High-Boost Mesh Filtering에서는 형상 강화를 위한 새로운 법선 벡터를 계산할 때 `면적 가중 평균`을 사용함
## 면적 가중 평균
<img width="446" height="162" alt="image" src="https://github.com/user-attachments/assets/9458c70f-02c9-409d-a7f2-d3e9c524f00e" />
<img width="454" height="246" alt="image" src="https://github.com/user-attachments/assets/7fee10f2-1468-4bba-8aec-73983afe1a4d" />

Figure 1. Normal-based error minimization 

`n(U)` : normal vector(법선 벡터) <br>
`A(U)` : face의 넓이 <br>
`$n^(k)(T)$` : k번 평활화한 smoothed normal <br>
`m(T)` : boosted normal(강화된 법선 벡터) <br>

## 💡연구의 필요성
면적 가중 평균 연산은 면적만을 가중치로 고려해 평균을 냄

- 연속적인 곡면을 완벽하게 표현하는 것이 어려움
- 복잡한 곡면이나 모서리와 같은 곡률의 변화가 급격한 경계 영역에서 계단현상 자주 발생 

따라서, Mesh의 **영역적 특성을 고려**한 방식으로의 개선이 필요함

# 제안 기법

# 결과
