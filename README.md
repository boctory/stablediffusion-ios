# StableDiffusion iOS App

## 프로젝트 소개
이 프로젝트는 Stable Diffusion을 활용하여 텍스트 프롬프트로부터 이미지를 생성하는 iOS 앱입니다. 최신 Stable Diffusion XL 모델을 사용하여 고품질의 이미지를 생성할 수 있습니다.

## 주요 기능
- 텍스트 프롬프트 입력을 통한 이미지 생성
- 생성된 이미지 저장 및 공유
- 이미지 생성 히스토리 관리
- 오프라인 모드 지원 (CoreML 모델 사용)

## 기술 스택
- Swift UI
- Core ML
- Stable Diffusion XL
- Combine Framework
- Swift Package Manager

## 요구사항
- iOS 16.0 이상
- Xcode 15.0 이상
- 최소 4GB 이상의 메모리를 가진 기기

## 설치 방법
1. 저장소 클론
```bash
git clone https://github.com/boctory/stablediffusion-ios.git
```

2. 의존성 설치
```bash
cd stablediffusion-ios
pod install
```

3. Xcode에서 프로젝트 열기
```bash
open StableDiffusion.xcworkspace
```

## 프로젝트 구조
```
StableDiffusion/
├── App/
│   ├── StableDiffusionApp.swift
│   └── AppDelegate.swift
├── Views/
│   ├── ContentView.swift
│   ├── PromptInputView.swift
│   └── ImageGenerationView.swift
├── Models/
│   ├── StableDiffusionModel.swift
│   └── ImageGenerator.swift
├── Services/
│   └── MLModelService.swift
└── Resources/
    └── stable-diffusion-xl-base.mlmodel
```

## 사용 방법
1. 앱을 실행합니다.
2. 텍스트 입력 필드에 원하는 이미지에 대한 설명을 입력합니다.
3. "생성하기" 버튼을 탭하여 이미지를 생성합니다.
4. 생성된 이미지를 저장하거나 공유할 수 있습니다.

## 문제 해결
- 이미지 생성이 느린 경우: 기기의 성능에 따라 이미지 생성 시간이 달라질 수 있습니다.
- 메모리 부족 오류: 앱을 재시작하거나 기기를 재부팅해보세요.
- 모델 다운로드 실패: 네트워크 연결을 확인하고 다시 시도해보세요.

## 라이선스
MIT License

## 기여 방법
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 