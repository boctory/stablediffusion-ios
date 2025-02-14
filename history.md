# Stable Diffusion 프로젝트 개발 히스토리

## 1. iOS 앱 개발

### 초기 설정
- XcodeGen을 사용하여 프로젝트 구조 설정
- iOS 18.1 타겟으로 설정
- Xcode 16.2 버전 지정
- 필요한 Info.plist 키 설정
- 코드 서명 설정 구성

### 핵심 컴포넌트 구현
1. **파이프라인 구현** (`StableDiffusionPipeline.swift`)
   - CoreML 모델 통합
   - 텍스트-이미지 생성 로직 구현
   - 비동기 처리 최적화

2. **스케줄러 개발** (`DDIMScheduler.swift`)
   - DDIM 스케줄링 알고리즘 구현
   - 노이즈 제거 프로세스 최적화

3. **토크나이저 구현** (`CLIPTokenizer.swift`)
   - CLIP 토크나이저 통합
   - 텍스트 인코딩 처리

4. **UI 개발** (`ContentView.swift`)
   - SwiftUI를 사용한 사용자 인터페이스 구현
   - 이미지 생성 프로세스 상태 표시
   - 스타일 선택 기능 구현

### 테스트
- 단위 테스트 구현 (`StableDiffusionTests.swift`)
- 토크나이저, 스케줄러, 파이프라인 테스트 케이스 작성

## 2. 웹 버전 개발

### 웹 인터페이스 구현
- HTML/CSS/JavaScript 기반 웹 앱 구현
- Tailwind CSS를 활용한 모던 UI 디자인
- 반응형 레이아웃 구현

### 기능 구현
- 텍스트 프롬프트 입력
- 이미지 스타일 선택 (사진, 디지털 아트, 애니메이션, 만화)
- API 통합 및 이미지 생성 처리
- 로딩 상태 및 에러 처리

### 보안 설정
- API 키 관리를 위한 설정 파일 분리
- config.js 파일을 통한 API 키 관리
- .gitignore 설정으로 민감한 정보 보호

## 3. 버전 관리

### GitHub 저장소 설정
1. iOS 버전:
   - https://github.com/boctory/stablediffusion-ios
   - 핵심 소스 코드만 포함
   - 큰 모델 파일 제외

2. 웹 버전:
   - 로컬 웹 서버 구현 (포트 8000)
   - 웹 인터페이스 파일 관리

## 4. 개발 환경
- Python 가상 환경 설정
- CoreML 모델 변환 스크립트 구현
- 의존성 관리 (requirements.txt)

## 5. 향후 개선 사항
- 모델 다운로드 및 변환 자동화
- 성능 최적화
- 추가 이미지 스타일 지원
- 배치 처리 기능 구현 